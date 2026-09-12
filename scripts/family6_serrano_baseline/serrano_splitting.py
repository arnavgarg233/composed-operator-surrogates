"""Serrano et al. (arXiv:2602.00884) test-time neural operator splitting, as a
same-budget baseline for the composed-operator-surrogates project.

Everything this script does is fixed in METHOD_SPEC.md, which was written before any
model here was fitted. Read that first; this file is its implementation and nothing
more. In particular the equal-budget contract (METHOD_SPEC section 4), the metric
definitions (section 5), and the gates (section 6) live there, not here.

PROVENANCE
----------
The package is never modified. Its harness is *imported* so that the code that fits a
model here is byte-for-byte the code that fits a model there:

    dependency/scripts/baseline/timing_probe.py  -> tp
        physics constants, build_steppers/initial_conditions/rollout,
        init_fno/fno_apply, adam_init/adam_update/train_step
    dependency/scripts/baseline/run_baseline.py  -> rb
        K, STEPS, BATCH, LR_HIGH/LR_LOW, NARROW, BROAD, relative_l2, make_pairs,
        train, and the gate thresholds
    dependency/scripts/band_check/_runtime_path.py
        pinned runtime on sys.path, imported transitively by timing_probe

Vendored (copied verbatim, attributed at the point of use) rather than imported,
because dependency/scripts/baseline/run_dissociation.py parses sys.argv and calls
mkdir at import time and importing it would be a write into a read-only tree:

    run_dissociation.py:centred_ratio         -> centred_ratio   (the S metric)
    run_dissociation.py:compose_predicted     -> compose_predicted
    run_dissociation.py:compose_true          -> compose_true
    run_dissociation.py:APPLICATIONS_PER_LEG  -> APPLICATIONS_PER_LEG

New code, all of it: the operator/splitting abstraction (Lie, Strang, Strang at grain
2 tau), the test-time composition search (Algorithm 1 beam, Algorithm 2 uniform,
plus exhaustive), the context constructions, and the arm bookkeeping.

Environment (threads pinned, CPU):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    .venv/bin/python serrano_splitting.py <subcmd>

Subcommands
    selftest                          gates U1 and U2, float64, no fitting, seconds
    fit <cond> <primitive> <seed>     one A1 FNO -> ckpt/<cond>_<primitive>_seed<k>.npz
    evaluate <seed>                   all arms for one seed -> results/seed<k>.json
    aggregate                         -> results/RESULT_SMOKE.json
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
PKG = RELEASE_ROOT / "dependency"
sys.path.insert(0, str(PKG / "scripts" / "baseline"))
sys.path.insert(0, str(PKG / "scripts" / "band_check"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import timing_probe as tp  # noqa: E402
import run_baseline as rb  # noqa: E402

CKPT = RELEASE_ROOT / "work" / "family6_serrano_baseline" / "ckpt"
RESULTS = RELEASE_ROOT / "results" / "family6_serrano_baseline"

# ---------------------------------------------------------------------------
# Constants. Every one of these is either imported from the package above or
# declared in METHOD_SPEC.md. Nothing is invented here.
# ---------------------------------------------------------------------------
K = rb.K                              # 10; one application advances tau = 0.10
TAU = K * tp.DT                       # 0.10
APPLICATIONS_PER_LEG = 10             # run_dissociation.py:APPLICATIONS_PER_LEG
FRAMES = 2 * APPLICATIONS_PER_LEG + 1  # 21
SMOKE_UNITS = 48                      # TASK.md step 3
MASTER = tp.MASTER_SEED               # 20260905

CONTEXT_FRAMES = 7                    # METHOD_SPEC 3(f); 6 transitions
MAX_LENGTH = 5                        # paper section 4.2, M = 5
BEAM_WIDTH = 4                        # paper section 4.2, B = 4
BEAM_MIN_REL_IMPROVEMENT = 0.01       # METHOD_SPEC 3(e), open choice
UNIFORM_TRIALS = 100                  # paper section 4.2, T = 100
UNIFORM_SEED = 20260910

PRIMITIVES = ("diffusion", "reaction")
CONDITIONS = ("narrow", "broad")
SUPPORTS = {"narrow": rb.NARROW, "broad": rb.BROAD}

# Evaluation-cohort seeds. +7/+8 are run_baseline.py's on/switch seeds, +11 is
# run_dissociation.py's eval seed. +21/+22 are new and disjoint (METHOD_SPEC 5.1).
SEED_ON, SEED_SWITCH, SEED_S, SEED_CAL_ON, SEED_CAL_OFF = 7, 8, 11, 21, 22


# ---------------------------------------------------------------------------
# The evaluation forward pass.
#
# tp.fno_apply is a plain function; the package calls it uncompiled, which costs
# roughly one XLA dispatch per primitive op per call. The evaluation sweep here
# makes tens of thousands of such calls, so it is wrapped in jax.jit -- the SAME
# function, compiled. Training is untouched (tp.train_step was already jitted in
# the package). Every arm, ours and the baseline's, goes through this one path,
# and evaluate() records a numerical agreement check against the uncompiled
# function in the result file.
# ---------------------------------------------------------------------------
FNO_FORWARD = jax.jit(tp.fno_apply)


# ===========================================================================
# Vendored from dependency/scripts/baseline/run_dissociation.py, verbatim.
# ===========================================================================
def centred_ratio(predicted: np.ndarray, true: np.ndarray) -> float:
    """run_dissociation.py:centred_ratio -- the S metric.

    Learner error on the unit-specific order field, over a no-structure predictor.
    """
    centred_true = true - true.mean(axis=0, keepdims=True)
    centred_pred = predicted - predicted.mean(axis=0, keepdims=True)
    return float(np.sqrt(((centred_pred - centred_true) ** 2).mean())
                 / np.sqrt((centred_true ** 2).mean()))


def compose_predicted(first, second, start, grid) -> np.ndarray:
    """run_dissociation.py:compose_predicted -- the oracle-schedule arm.

    Apply one surrogate ten times, then the other ten, keeping every frame.
    Verbatim but for FNO_FORWARD in place of tp.fno_apply; see above.
    """
    frames, state = [start], start
    for params in (first, second):
        for _ in range(APPLICATIONS_PER_LEG):
            state = FNO_FORWARD(params, state, grid)
            frames.append(state)
    return np.asarray(jnp.stack(frames, axis=1), dtype=np.float64).squeeze(2)


def compose_true(first, second, start) -> np.ndarray:
    """run_dissociation.py:compose_true."""
    frames, state = [start], start
    for stepper in (first, second):
        for _ in range(APPLICATIONS_PER_LEG):
            state = tp.rollout(stepper, state, K)[:, -1]
            frames.append(state)
    return np.asarray(jnp.stack(frames, axis=1), dtype=np.float64).squeeze(2)


# ===========================================================================
# Operators and splitting. NEW CODE.
#
# An operator carries a full step and, when one exists, a half step. Serrano et
# al.'s dictionary members are continuous-time vector fields, so a half step is
# free for them; an A1 FNO is a discrete tau-map and has none, which is why
# METHOD_SPEC 3(c) runs Strang at grain 2 tau instead.
# ===========================================================================
class Operator:
    __slots__ = ("name", "full", "half")

    def __init__(self, name, full, half=None):
        self.name, self.full, self.half = name, full, half

    def doubled(self) -> "Operator":
        """Grain 2 tau: the native map becomes the half step (METHOD_SPEC 3(c))."""
        return Operator(self.name + "^2", lambda u: self.full(self.full(u)), self.full)


def lie_apply(ops, seq, u):
    """Paper section 4.3: u^{L+1} = f_{i_m} o ... o f_{i_1}(u^L)."""
    for i in seq:
        u = ops[i].full(u)
    return u


def strang_apply(ops, seq, u):
    """Paper section 4.3: the palindromic symmetric composition.

    For m = 2 this is f_1^{dt/2} o f_2^{dt} o f_1^{dt/2}, the paper's displayed form.
    For m = 1 it degenerates to the full step, as it must.
    """
    for i in seq[:-1]:
        u = ops[i].half(u)
    u = ops[seq[-1]].full(u)
    for i in reversed(seq[:-1]):
        u = ops[i].half(u)
    return u


def candidates(num_ops: int, max_length: int = MAX_LENGTH):
    """Ordered multisets of length 1..M. METHOD_SPEC 3(a)/(b)."""
    out = []
    for m in range(1, max_length + 1):
        out.extend(itertools.product(range(num_ops), repeat=m))
    return out


# ===========================================================================
# Test-time composition search. NEW CODE.
#
# Every selection rule reads the same per-unit loss table, so beam and uniform
# cost no forward passes beyond the exhaustive sweep that builds it. This is an
# implementation convenience only: the three rules select exactly what they would
# select if run independently.
# ===========================================================================
def loss_table(ops, cands, context, apply_fn) -> np.ndarray:
    """L(S) per candidate per unit. Paper section 4.2, teacher-forced one step.

    context: (units, L, 1, N) at the splitting grain. Returns (len(cands), units).
    """
    ctx = jnp.asarray(context)
    units, length, npoints = ctx.shape[0], ctx.shape[1], ctx.shape[-1]
    # All L-1 teacher-forced transitions of all units in one batch.
    flat = ctx[:, :-1].reshape(-1, 1, npoints)
    target = ctx[:, 1:].reshape(-1, 1, npoints)
    den = jnp.sqrt(jnp.sum(target ** 2, axis=(1, 2)))

    def score(pred):
        num = jnp.sqrt(jnp.sum((pred - target) ** 2, axis=(1, 2)))
        return np.asarray(num / den, np.float64).reshape(units, length - 1).mean(axis=1)

    table = np.empty((len(cands), units), dtype=np.float64)
    if apply_fn is lie_apply:
        # Lie compositions share prefixes, so the whole sweep costs one forward
        # pass per candidate instead of one per operator in it. candidates() is
        # ordered by length, so every prefix is already cached when it is needed.
        cache = {(): flat}
        for c, seq in enumerate(cands):
            pred = cache.get(seq)
            if pred is None:
                pred = ops[seq[-1]].full(cache[seq[:-1]])
                cache[seq] = pred
            table[c] = score(pred)
    else:
        for c, seq in enumerate(cands):
            table[c] = score(apply_fn(ops, seq, flat))
    return table


def select_exhaustive(table, cands):
    return [cands[i] for i in np.argmin(table, axis=0)]


def select_beam(table, cands, num_ops, beam=BEAM_WIDTH, max_length=MAX_LENGTH,
                threshold=BEAM_MIN_REL_IMPROVEMENT):
    """Algorithm 1, Beam Search Operator Composition, per unit."""
    index = {seq: i for i, seq in enumerate(cands)}
    chosen = []
    for u in range(table.shape[1]):
        def loss(seq):
            return table[index[seq], u]
        singles = sorted((tuple([j]) for j in range(num_ops)), key=loss)
        current = singles[:beam]
        best, best_loss = current[0], loss(current[0])
        for _ in range(max_length - 1):
            expanded = sorted(
                (s + (j,) for s in current for j in range(num_ops)), key=loss)
            current = expanded[:beam]
            head_loss = loss(current[0])
            if head_loss >= best_loss * (1.0 - threshold):
                break                       # minimum relative improvement not met
            best, best_loss = current[0], head_loss
        chosen.append(best)
    return chosen


def select_uniform(table, cands, num_ops, trials=UNIFORM_TRIALS,
                   max_length=MAX_LENGTH, seed=UNIFORM_SEED):
    """Algorithm 2, Uniform Operator Composition Search."""
    rng = np.random.default_rng(seed)
    index = {seq: i for i, seq in enumerate(cands)}
    rows = []
    for _ in range(trials):
        m = int(rng.integers(1, max_length + 1))
        rows.append(index[tuple(int(x) for x in rng.integers(0, num_ops, size=m))])
    rows = np.asarray(sorted(set(rows)))
    sub = table[rows]
    return [cands[rows[i]] for i in np.argmin(sub, axis=0)]


def select_all(table, cands, num_ops):
    return {
        "exhaustive": select_exhaustive(table, cands),
        "beam": select_beam(table, cands, num_ops),
        "uniform": select_uniform(table, cands, num_ops),
    }


def program(seq, apply_fn):
    """The flat (operator, step-kind) schedule a composition executes.

    Lie is the sequence itself at full steps; Strang is the palindrome of half
    steps around the innermost full step. Same semantics as lie_apply /
    strang_apply, written as data so it can be applied position by position.
    """
    if apply_fn is lie_apply:
        return [(i, "full") for i in seq]
    return ([(i, "half") for i in seq[:-1]] + [(seq[-1], "full")]
            + [(i, "half") for i in reversed(seq[:-1])])


def _apply_programs(ops, programs, state):
    """Apply a per-unit program to its own row of a (units, 1, N) batch.

    Vectorised by POSITION, not grouped by sequence: at each position at most
    one forward pass per distinct (operator, step-kind) still in play, on the
    whole batch. Cost is therefore bounded by depth x 2|D| forward passes however
    diverse the per-unit selections are. Grouping by sequence instead degenerates
    to batch-of-one passes when every unit picks a different composition, which
    is exactly what an untrained or weakly-trained dictionary produces.
    """
    depth = max((len(p) for p in programs), default=0)
    for pos in range(depth):
        need: dict = {}
        for row, prog in enumerate(programs):
            if pos < len(prog):
                need.setdefault(prog[pos], []).append(row)
        if not need:
            break
        updated = np.asarray(state).copy()
        for (op_index, kind), rows in need.items():
            fn = ops[op_index].full if kind == "full" else ops[op_index].half
            out = np.asarray(fn(state))
            updated[np.asarray(rows)] = out[np.asarray(rows)]
        state = jnp.asarray(updated)
    return state


def apply_selected(ops, selected, u, apply_fn):
    """Apply a per-unit composition once to a (units, 1, N) batch."""
    return _apply_programs(ops, [program(s, apply_fn) for s in selected], u)


def rollout_selected(ops, selected, u0, apply_fn, macro_steps):
    """Autoregressive rollout with a per-unit composition. Returns (units, F, N)."""
    programs = [program(s, apply_fn) for s in selected]
    state = jnp.asarray(u0)
    frames = np.empty((state.shape[0], macro_steps + 1, state.shape[-1]),
                      dtype=np.float64)
    frames[:, 0] = np.asarray(state, np.float64).squeeze(1)
    for j in range(macro_steps):
        state = _apply_programs(ops, programs, state)
        frames[:, j + 1] = np.asarray(state, np.float64).squeeze(1)
    return frames


# ===========================================================================
# Checkpoints. NEW CODE.
# ===========================================================================
_LIST_KEYS = ("spectral_re", "spectral_im", "local_w", "local_b")


def save_params(path: Path, params) -> None:
    flat = {}
    for key, value in params.items():
        if key in _LIST_KEYS:
            for i, a in enumerate(value):
                flat[f"{key}__{i}"] = np.asarray(a)
        else:
            flat[key] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat)


def load_params(path: Path):
    raw = np.load(path)
    params = {k: jnp.asarray(raw[k]) for k in raw.files if "__" not in k}
    for key in _LIST_KEYS:
        items = sorted((k for k in raw.files if k.startswith(key + "__")),
                       key=lambda k: int(k.split("__")[-1]))
        params[key] = [jnp.asarray(raw[k]) for k in items]
    return params


# ===========================================================================
# selftest: gates U1 and U2 (METHOD_SPEC section 6). float64, no fitting.
# ===========================================================================
def analytic_diffusion(diffusivity: float, dt: float, num_points: int):
    """exp(-D k^2 dt) as a Fourier multiplier. Diagonal, hence commuting."""
    kx = 2.0 * np.pi * np.fft.rfftfreq(num_points, d=1.0 / num_points)

    def step(u, factor=1.0):
        c = jnp.fft.rfft(jnp.asarray(u), axis=-1)
        mult = jnp.asarray(np.exp(-diffusivity * kx ** 2 * dt * factor))
        return jnp.fft.irfft(c * mult, n=num_points, axis=-1)

    return Operator(f"D{diffusivity}", lambda u: step(u, 1.0), lambda u: step(u, 0.5))


def selftest() -> int:
    jax.config.update("jax_enable_x64", True)
    started = time.perf_counter()
    n, dt = tp.NUM_POINTS, TAU
    d1, d2 = 0.004, 0.006
    op1, op2 = analytic_diffusion(d1, dt, n), analytic_diffusion(d2, dt, n)
    exact = analytic_diffusion(d1 + d2, dt, n)
    exact2 = analytic_diffusion(d1 + d2, 2 * dt, n)
    ops = [op1, op2]

    key = jr.PRNGKey(0)
    u0 = jnp.asarray(
        np.asarray(jr.normal(key, (16, 1, n)), np.float64) * 0.175 + 0.5)

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.max(np.sqrt(((a - b) ** 2).sum(axis=(1, 2)))
                            / np.sqrt((b ** 2).sum(axis=(1, 2)))))

    target1 = exact.full(u0)
    errors = {
        "lie_12": rel(lie_apply(ops, (0, 1), u0), target1),
        "lie_21": rel(lie_apply(ops, (1, 0), u0), target1),
        "strang_native_12": rel(strang_apply(ops, (0, 1), u0), target1),
        "strang_native_21": rel(strang_apply(ops, (1, 0), u0), target1),
    }
    doubled = [op1.doubled(), op2.doubled()]
    target2 = exact2.full(u0)
    errors["strang_2tau_12"] = rel(strang_apply(doubled, (0, 1), u0), target2)
    errors["strang_2tau_21"] = rel(strang_apply(doubled, (1, 0), u0), target2)
    # A three-operator palindrome, to exercise the general form.
    third = analytic_diffusion(0.002, dt, n)
    errors["strang_native_123"] = rel(
        strang_apply([op1, op2, third], (0, 1, 2), u0),
        analytic_diffusion(d1 + d2 + 0.002, dt, n).full(u0))

    u1_floor = 1e-10
    u1 = max(errors.values()) <= u1_floor

    # U2: the search must recover the composition from a context the combined
    # operator generated.
    frames = [u0]
    for _ in range(CONTEXT_FRAMES - 1):
        frames.append(exact.full(frames[-1]))
    context = jnp.stack(frames, axis=1)
    cands = candidates(2)
    table = loss_table(ops, cands, context, lie_apply)
    picks = select_all(table, cands, 2)
    best_losses = {rule: float(np.max([table[cands.index(s), u]
                                       for u, s in enumerate(sel)]))
                   for rule, sel in picks.items()}
    both = {rule: all(0 in s and 1 in s and len(s) == 2 for s in sel)
            for rule, sel in picks.items()}
    u2 = (all(both[r] for r in ("exhaustive", "beam"))
          and max(best_losses[r] for r in ("exhaustive", "beam")) <= u1_floor)

    payload = {
        "schema": "family6-selftest-v1",
        "gate_U1": {"statement": "Lie, native-grain Strang and 2-tau-grain Strang are "
                                 "exact on a commuting pair",
                    "pair": {"D1": d1, "D2": d2, "dt": dt, "dtype": "float64"},
                    "max_relative_l2": errors, "floor": u1_floor, "pass": bool(u1)},
        "gate_U2": {"statement": "the search recovers the true composition from a "
                                 "context the combined operator generated",
                    "selected_unique": {r: sorted({tuple(s) for s in sel})
                                        for r, sel in picks.items()},
                    "worst_selected_loss": best_losses,
                    "all_units_length2_both_ops": both,
                    "floor": u1_floor, "pass": bool(u2)},
        "num_candidates": len(cands),
        "elapsed_seconds": time.perf_counter() - started,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SELFTEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nU1 {'PASS' if u1 else 'FAIL'}   U2 {'PASS' if u2 else 'FAIL'}")
    return 0 if (u1 and u2) else 2


# ===========================================================================
# fit
# ===========================================================================
def fit(condition: str, primitive: str, seed: int) -> None:
    if condition not in CONDITIONS or primitive not in PRIMITIVES:
        raise SystemExit(f"unknown arm {condition}/{primitive}")
    started = time.perf_counter()
    diffusion, reaction = tp.build_steppers(tp.DT)
    stepper = diffusion if primitive == "diffusion" else reaction
    grid = jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)

    # Equal budget: 640 units, the package's IC generator, the package's seed
    # derivation (run_baseline.py uses MASTER_SEED + 1 for training ICs).
    ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, SUPPORTS[condition], MASTER + 1)
    states = tp.rollout(stepper, ic, tp.STEPS_PER_TRAJECTORY)
    pairs_in, pairs_out = rb.make_pairs(states)
    params = rb.train(pairs_in, pairs_out, grid, seed)

    path = CKPT / f"{condition}_{primitive}_seed{seed}.npz"
    save_params(path, params)
    elapsed = time.perf_counter() - started
    meta = {"condition": condition, "primitive": primitive, "seed": seed,
            "train_units": int(tp.NUM_TRAIN_UNITS), "ic_offset_range": list(SUPPORTS[condition]),
            "steps": rb.STEPS, "batch": rb.BATCH, "k": K,
            "num_training_pairs": int(pairs_in.shape[0]),
            "elapsed_seconds": elapsed}
    (CKPT / f"{condition}_{primitive}_seed{seed}.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"fitted {condition}/{primitive} seed {seed} in {elapsed / 60:.1f} min "
          f"-> {path.name}")


# ===========================================================================
# evaluate
# ===========================================================================
def _fno_op(name, params, grid):
    return Operator(name, lambda u: FNO_FORWARD(params, u, grid))


def _stride(traj, every=K):
    """(units, dt_frames, 1, N) -> the tau-stride subsequence."""
    return traj[:, ::every]


def evaluate(seed: int) -> None:
    started = time.perf_counter()
    _t = [time.perf_counter()]

    def stage(label: str) -> None:
        now = time.perf_counter()
        print(f"    [{now - started:7.1f}s] {label} (+{now - _t[0]:.1f}s)", flush=True)
        _t[0] = now

    diffusion, reaction = tp.build_steppers(tp.DT)
    grid = jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)
    models = {}
    for condition in CONDITIONS:
        for primitive in PRIMITIVES:
            path = CKPT / f"{condition}_{primitive}_seed{seed}.npz"
            if not path.exists():
                raise SystemExit(f"missing checkpoint {path}")
            models[(condition, primitive)] = load_params(path)

    probe = tp.initial_conditions(8, rb.NARROW, MASTER + 99)
    _p = models[("narrow", "diffusion")]
    _grid = jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)
    _a = np.asarray(FNO_FORWARD(_p, probe, _grid), np.float64)
    _b = np.asarray(tp.fno_apply(_p, probe, _grid), np.float64)
    jit_agreement = float(np.max(np.abs(_a - _b)) / np.max(np.abs(_b)))
    stage("checkpoints loaded")
    result: dict = {
        "schema": "family6-serrano-smoke-v1",
        "seed": seed,
        "units": SMOKE_UNITS,
        "num_points": tp.NUM_POINTS,
        "tau": TAU,
        "k": K,
        "context_frames": CONTEXT_FRAMES,
        "max_length": MAX_LENGTH,
        "beam_width": BEAM_WIDTH,
        "uniform_trials": UNIFORM_TRIALS,
        "platform": {"python": sys.version.split()[0], "machine": platform.machine(),
                     "jax": jax.__version__},
        "jit_vs_uncompiled_max_relative_difference": jit_agreement,
    }

    # ---------------------------------------------------------------- R metric
    on_ic = tp.initial_conditions(SMOKE_UNITS, rb.NARROW, MASTER + SEED_ON)
    switch_ic = tp.initial_conditions(SMOKE_UNITS, rb.NARROW, MASTER + SEED_SWITCH)
    on_traj = tp.rollout(diffusion, on_ic, tp.STEPS_PER_TRAJECTORY)      # 101 frames
    switch_traj = tp.rollout(reaction, switch_ic, tp.STEPS_PER_TRAJECTORY)

    x_on = on_traj[:, -1]                       # frame 10 at tau stride, t = 1.0
    x_off = switch_traj[:, -1]                  # the package's switch state
    x_on0 = on_traj[:, 0]                       # package-native bridge state
    y_on = tp.rollout(diffusion, x_on, K)[:, -1]
    y_off = tp.rollout(diffusion, x_off, K)[:, -1]
    y_on0 = tp.rollout(diffusion, x_on0, K)[:, -1]

    # Causal contexts: the 7 tau-frames ending at the evaluation state.
    ctx_on_causal = _stride(on_traj)[:, -CONTEXT_FRAMES:]
    ctx_off_causal = _stride(switch_traj)[:, -CONTEXT_FRAMES:]

    # Regime-matched (oracle) contexts, from disjoint index-matched cohorts.
    cal_on_ic = tp.initial_conditions(SMOKE_UNITS, rb.NARROW, MASTER + SEED_CAL_ON)
    cal_on_traj = tp.rollout(diffusion, cal_on_ic, tp.STEPS_PER_TRAJECTORY)
    ctx_on_regime = _stride(cal_on_traj)[:, -CONTEXT_FRAMES:]
    cal_off_ic = tp.initial_conditions(SMOKE_UNITS, rb.NARROW, MASTER + SEED_CAL_OFF)
    cal_switch = tp.rollout(reaction, cal_off_ic, tp.STEPS_PER_TRAJECTORY)[:, -1]
    cal_leg2 = tp.rollout(diffusion, cal_switch, K * (CONTEXT_FRAMES - 1))
    ctx_off_regime = _stride(cal_leg2)[:, :CONTEXT_FRAMES]

    def med_rel(pred, target):
        return float(np.median(rb.relative_l2(np.asarray(pred, np.float64),
                                              np.asarray(target, np.float64))))

    # Support geometry (G1) and persistence (G2), data properties.
    train_ic_narrow = tp.initial_conditions(tp.NUM_TRAIN_UNITS, rb.NARROW, MASTER + 1)
    train_frames = tp.rollout(diffusion, train_ic_narrow, tp.STEPS_PER_TRAJECTORY)
    train_means = np.asarray(train_frames.mean(axis=(2, 3))).reshape(-1)
    switch_means = np.asarray(x_off.mean(axis=(1, 2))).reshape(-1)
    support = (float(train_means.min()), float(train_means.max()))
    switch_support = (float(switch_means.min()), float(switch_means.max()))
    overlap = max(0.0, min(support[1], switch_support[1])
                  - max(support[0], switch_support[0]))
    persistence_on = med_rel(x_on, y_on)
    persistence_off = med_rel(x_off, y_off)
    persistence_on0 = med_rel(x_on0, y_on0)

    stage("R corpora + support/persistence")
    r_arms: dict = {}
    cands = candidates(2)

    for condition in CONDITIONS:
        op_diff = _fno_op("diff", models[(condition, "diffusion")], grid)
        r_arms[f"{condition}_oracle"] = {
            "kind": "oracle-schedule (told the primitive)", "context": "none",
            "extra_test_time_forward_passes_per_unit": 0,
            "e_on": med_rel(op_diff.full(x_on), y_on),
            "e_off": med_rel(op_diff.full(x_off), y_off),
            "e_on_frame0_bridge": med_rel(op_diff.full(x_on0), y_on0),
            "e_off_frame0_bridge": med_rel(op_diff.full(x_off), y_off),
        }
        ops = [op_diff, _fno_op("reac", models[(condition, "reaction")], grid)]
        for ctx_name, c_on, c_off in (
            ("causal", ctx_on_causal, ctx_off_causal),
            ("regime", ctx_on_regime, ctx_off_regime),
        ):
            if condition == "broad" and ctx_name == "regime":
                continue                      # METHOD_SPEC 5.3: not an arm
            t_on = loss_table(ops, cands, c_on, lie_apply)
            t_off = loss_table(ops, cands, c_off, lie_apply)
            sel_on, sel_off = select_all(t_on, cands, 2), select_all(t_off, cands, 2)
            entry = {
                "kind": "searched (Lie)",
                "context": ctx_name + (" [ORACLE]" if ctx_name == "regime" else ""),
                "extra_test_time_forward_passes_per_unit":
                    int(sum(len(s) for s in cands) * (CONTEXT_FRAMES - 1)),
                "by_rule": {},
            }
            for rule in ("exhaustive", "beam", "uniform"):
                p_on = apply_selected(ops, sel_on[rule], x_on, lie_apply)
                p_off = apply_selected(ops, sel_off[rule], x_off, lie_apply)
                e_on, e_off = med_rel(p_on, y_on), med_rel(p_off, y_off)
                entry["by_rule"][rule] = {
                    "e_on": e_on, "e_off": e_off, "R": e_off / e_on,
                    "selected_on": _hist(sel_on[rule]),
                    "selected_off": _hist(sel_off[rule]),
                    "diffusion_only_fraction_on": _diff_only(sel_on[rule]),
                }
            r_arms[f"serrano_{condition}_lie_{ctx_name}"] = entry
            stage(f"R search {condition}/{ctx_name}")

    for name, arm in r_arms.items():
        if "e_on" in arm:
            arm["R"] = arm["e_off"] / arm["e_on"]
            arm["R_frame0_bridge"] = (arm["e_off_frame0_bridge"]
                                      / arm["e_on_frame0_bridge"])

    result["R"] = {
        "estimand": f"k={K} diffusion solution operator, tau={TAU:.2f}",
        "evaluation_state": "frame 10 (t=1.0) of a one-leg trajectory; see "
                            "METHOD_SPEC 5.1 -- NOT comparable to the banked "
                            "110.234 / 17.650, which are at frame 0",
        "G1_shift_exists": {"training_support": list(support),
                            "switch_support": list(switch_support),
                            "overlap": overlap,
                            "gap": switch_support[0] - support[1],
                            "pass": bool(overlap == 0.0)},
        "persistence_on": persistence_on, "persistence_off": persistence_off,
        "persistence_on_frame0": persistence_on0,
        "arms": r_arms,
    }

    stage("R metric done")
    # ---------------------------------------------------------------- S metric
    s_ic = tp.initial_conditions(SMOKE_UNITS, rb.NARROW, MASTER + SEED_S)
    true_ab = compose_true(diffusion, reaction, s_ic)
    true_ba = compose_true(reaction, diffusion, s_ic)
    true_contrast = true_ab - true_ba
    stage("S truth (compose_true x2)")
    even = np.arange(0, FRAMES, 2)
    horizon = np.arange(CONTEXT_FRAMES, FRAMES)

    def s_scores(pred_ab, pred_ba):
        pc = pred_ab - pred_ba
        return {
            "trajectory": centred_ratio(pc, true_contrast),
            "endpoint": centred_ratio(pc[:, -1:], true_contrast[:, -1:]),
            "trajectory_even_subgrid": centred_ratio(pc[:, even],
                                                     true_contrast[:, even]),
            "endpoint_even_subgrid": centred_ratio(pc[:, -1:], true_contrast[:, -1:]),
            "trajectory_horizon_only": centred_ratio(pc[:, horizon],
                                                     true_contrast[:, horizon]),
        }

    s_arms: dict = {}
    for condition in CONDITIONS:
        p_diff, p_reac = models[(condition, "diffusion")], models[(condition, "reaction")]
        pred_ab = compose_predicted(p_diff, p_reac, s_ic, grid)
        pred_ba = compose_predicted(p_reac, p_diff, s_ic, grid)
        s_arms[f"{condition}_oracle"] = {
            "kind": "oracle-schedule (told the order). METHOD_SPEC 5.2: this is also "
                    "the oracle upper bound for the Serrano method on S.",
            "context": "none", "S": s_scores(pred_ab, pred_ba)}

        ops = [_fno_op("diff", p_diff, grid), _fno_op("reac", p_reac, grid)]
        ctx_ab = jnp.asarray(true_ab[:, :CONTEXT_FRAMES, None, :])
        ctx_ba = jnp.asarray(true_ba[:, :CONTEXT_FRAMES, None, :])
        t_ab = loss_table(ops, cands, ctx_ab, lie_apply)
        t_ba = loss_table(ops, cands, ctx_ba, lie_apply)
        sel_ab, sel_ba = select_all(t_ab, cands, 2), select_all(t_ba, cands, 2)
        entry = {"kind": "searched (Lie), time-homogeneous rollout",
                 "context": f"causal, first {CONTEXT_FRAMES} frames of the true "
                            f"composed trajectory (inside leg A)",
                 "by_rule": {}}
        for rule in ("exhaustive", "beam", "uniform"):
            p_ab = rollout_selected(ops, sel_ab[rule], s_ic, lie_apply,
                                    2 * APPLICATIONS_PER_LEG)
            p_ba = rollout_selected(ops, sel_ba[rule], s_ic, lie_apply,
                                    2 * APPLICATIONS_PER_LEG)
            entry["by_rule"][rule] = {"S": s_scores(p_ab, p_ba),
                                      "selected_ab": _hist(sel_ab[rule]),
                                      "selected_ba": _hist(sel_ba[rule])}
        s_arms[f"serrano_{condition}_lie_causal"] = entry
        stage(f"S search {condition} lie")

        if condition == "narrow":
            # Strang at grain 2 tau (METHOD_SPEC 3(c)). Context transitions must be
            # 2 tau apart, so the objective sees frames 0,2,4,6 -> 3 transitions.
            dops = [o.doubled() for o in ops]
            t_ab2 = loss_table(dops, cands, ctx_ab[:, ::2], strang_apply)
            t_ba2 = loss_table(dops, cands, ctx_ba[:, ::2], strang_apply)
            s_ab2, s_ba2 = select_all(t_ab2, cands, 2), select_all(t_ba2, cands, 2)
            entry = {"kind": "searched (Strang at grain 2 tau); even-subgrid only",
                     "context": f"causal, frames 0,2,4,6 of the true composed "
                                f"trajectory", "by_rule": {}}
            for rule in ("exhaustive", "beam", "uniform"):
                r_ab = rollout_selected(dops, s_ab2[rule], s_ic, strang_apply,
                                        APPLICATIONS_PER_LEG)
                r_ba = rollout_selected(dops, s_ba2[rule], s_ic, strang_apply,
                                        APPLICATIONS_PER_LEG)
                pc = r_ab - r_ba
                tc = true_contrast[:, even]
                entry["by_rule"][rule] = {
                    "S": {"trajectory_even_subgrid": centred_ratio(pc, tc),
                          "endpoint_even_subgrid": centred_ratio(pc[:, -1:],
                                                                 tc[:, -1:])},
                    "selected_ab": _hist(s_ab2[rule]),
                    "selected_ba": _hist(s_ba2[rule])}
            s_arms["serrano_narrow_strang2tau_causal"] = entry
            stage("S search narrow strang2tau")

    result["S"] = {
        "definition": "run_dissociation.py:centred_ratio, centered over the 48 eval "
                      "units; trajectory = all 21 tau-frames, endpoint = frame 20",
        "frames": FRAMES, "applications_per_leg": APPLICATIONS_PER_LEG,
        "arms": s_arms,
    }

    stage("S metric done")
    # ---------------------------------------------------------------- gates
    e_ons = [r_arms[f"{c}_oracle"]["e_on"] for c in CONDITIONS]
    result["gates"] = {
        "G1_shift_exists": result["R"]["G1_shift_exists"],
        "G2_metric_resolves": {
            "dynamic_range": persistence_off / float(np.median(e_ons)),
            "required": rb.G2_MIN_DYNAMIC_RANGE,
            "pass": bool(persistence_off / float(np.median(e_ons))
                         >= rb.G2_MIN_DYNAMIC_RANGE)},
        "G4_censored_below": {"min_e_on": float(min(e_ons)), "floor": rb.G4_MIN_E_ON,
                              "pass": bool(min(e_ons) >= rb.G4_MIN_E_ON)},
        "P6_convergence": {"worst_e_on": float(max(e_ons)), "limit": rb.P6_MAX_E_ON,
                           "pass": bool(max(e_ons) <= rb.P6_MAX_E_ON)},
        "S1_search_sanity": {
            "statement": "on the on-support causal R context the search selects a "
                         "diffusion-only composition for >= 90% of units",
            "fraction": r_arms["serrano_narrow_lie_causal"]["by_rule"]
                        ["exhaustive"]["diffusion_only_fraction_on"],
            "required": 0.90,
            "pass": bool(r_arms["serrano_narrow_lie_causal"]["by_rule"]["exhaustive"]
                         ["diffusion_only_fraction_on"] >= 0.90)},
    }
    result["elapsed_seconds"] = time.perf_counter() - started

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"seed {seed} evaluated in {result['elapsed_seconds'] / 60:.1f} min")
    for name, arm in r_arms.items():
        if "R" in arm:
            print(f"  R  {name:36} {arm['R']:10.3f}")
        else:
            for rule, v in arm["by_rule"].items():
                print(f"  R  {name:28}/{rule:9} {v['R']:10.3f}")
    for name, arm in s_arms.items():
        if "S" in arm:
            print(f"  S  {name:36} traj {arm['S']['trajectory']:.4f} "
                  f"end {arm['S']['endpoint']:.4f}")
        else:
            for rule, v in arm["by_rule"].items():
                s = v["S"]
                print(f"  S  {name:28}/{rule:9} "
                      + "  ".join(f"{k} {x:.4f}" for k, x in s.items()))


def _hist(selected):
    out: dict = {}
    for seq in selected:
        key = "-".join("D" if i == 0 else "R" for i in seq)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _diff_only(selected):
    return float(np.mean([all(i == 0 for i in s) for s in selected]))


# ===========================================================================
# aggregate
# ===========================================================================
def aggregate() -> None:
    files = sorted(RESULTS.glob("seed*.json"))
    if not files:
        raise SystemExit("no per-seed results")
    per_seed = [json.loads(p.read_text()) for p in files]
    out: dict = {"schema": "family6-serrano-smoke-aggregate-v1",
                 "seeds": [d["seed"] for d in per_seed],
                 "units": per_seed[0]["units"],
                 "note": "2 seeds: ranges are printed, nothing is inferred from them "
                         "(METHOD_SPEC section 7.6).",
                 "R": {}, "S": {}, "gates": {}}

    def collect(path_fn):
        vals = []
        for d in per_seed:
            try:
                vals.append(path_fn(d))
            except (KeyError, TypeError):
                pass
        return vals

    for name in per_seed[0]["R"]["arms"]:
        arm = per_seed[0]["R"]["arms"][name]
        if "R" in arm:
            for field in ("e_on", "e_off", "R", "R_frame0_bridge"):
                vals = collect(lambda d, n=name, f=field: d["R"]["arms"][n][f])
                out["R"].setdefault(name, {})[field] = {
                    "median": float(np.median(vals)), "per_seed": vals}
        else:
            for rule in ("exhaustive", "beam", "uniform"):
                for field in ("e_on", "e_off", "R"):
                    vals = collect(lambda d, n=name, r=rule, f=field:
                                   d["R"]["arms"][n]["by_rule"][r][f])
                    out["R"].setdefault(f"{name}/{rule}", {})[field] = {
                        "median": float(np.median(vals)), "per_seed": vals}

    for name in per_seed[0]["S"]["arms"]:
        arm = per_seed[0]["S"]["arms"][name]
        if "S" in arm:
            for field in arm["S"]:
                vals = collect(lambda d, n=name, f=field: d["S"]["arms"][n]["S"][f])
                out["S"].setdefault(name, {})[field] = {
                    "median": float(np.median(vals)), "per_seed": vals}
        else:
            for rule in ("exhaustive", "beam", "uniform"):
                for field in arm["by_rule"][rule]["S"]:
                    vals = collect(lambda d, n=name, r=rule, f=field:
                                   d["S"]["arms"][n]["by_rule"][r]["S"][f])
                    out["S"].setdefault(f"{name}/{rule}", {})[field] = {
                        "median": float(np.median(vals)), "per_seed": vals}

    for gate in per_seed[0]["gates"]:
        out["gates"][gate] = {"per_seed_pass": [d["gates"][gate]["pass"]
                                                for d in per_seed],
                              "all_pass": all(d["gates"][gate]["pass"]
                                              for d in per_seed)}
    fit_meta = sorted(CKPT.glob("*.json"))
    out["fit_wall_clock_seconds"] = {p.stem: json.loads(p.read_text())["elapsed_seconds"]
                                     for p in fit_meta}
    out["fit_wall_clock_total_seconds"] = float(
        sum(out["fit_wall_clock_seconds"].values()))
    out["eval_wall_clock_seconds"] = {str(d["seed"]): d["elapsed_seconds"]
                                      for d in per_seed}
    (RESULTS / "RESULT_SMOKE.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["gates"], indent=2))
    print(f"\nwrote {RESULTS / 'RESULT_SMOKE.json'}")


def main(argv) -> None:
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "fit":
        fit(argv[1], argv[2], int(argv[3]))
    elif cmd == "evaluate":
        evaluate(int(argv[1]))
    elif cmd == "aggregate":
        aggregate()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
