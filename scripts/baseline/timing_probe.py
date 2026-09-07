"""Timing probe for the standard-architecture baseline. MEASURES ONLY.

This builds the real corpus slice at the real problem size and times a small number
of real training steps. It does NOT train a model and it does NOT run the experiment.
Its whole output is a wall-clock extrapolation plus the corpus support statistics,
which are a property of the data rather than of any fitted model.

Run:
    ~/.local/bin/python3.12 scripts/baseline/timing_probe.py

Writes results/tables/baseline/TIMING_PROBE.json.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "band_check"))

import _runtime_path  # noqa: F401,E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import exponax as ex  # noqa: E402

# ----------------------------------------------------------------------------
# Physical configuration. Identical to recovery/band_check/band_check.py, which is
# itself recovered verbatim from S1. Nothing here is a new choice.
# ----------------------------------------------------------------------------
NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS = 1, 1.0, 256
DT = 0.01
DIFFUSIVITY = 0.01
REACTION_DIFFUSIVITY, REACTIVITY = 0.0, 1.0
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0

FRAMES_PER_TRAJECTORY = 101          # u0 plus 100 steps, tau = 1.00
STEPS_PER_TRAJECTORY = FRAMES_PER_TRAJECTORY - 1
NUM_TRAIN_UNITS = 640
NUM_SWITCH_UNITS = 640

# Narrow (training) support, recovered from the STATE_SUPPORT_SHIFT diagnosis.
IC_AMPLITUDE = 0.175
NARROW_OFFSET_RANGE = (0.4865, 0.5171)
# Reaction-produced switch-state spatial means, same source.
RECORDED_SWITCH_MEAN_RANGE = (0.6973, 0.7258)
IC_CUTOFF = 5                        # band_check reported 3, 5, 7; 5 is the midpoint
MASTER_SEED = 20260905

# ----------------------------------------------------------------------------
# FNO configuration. Li et al. 2021 FNO-1d defaults.
# ----------------------------------------------------------------------------
FNO_MODES = 16
FNO_WIDTH = 64
FNO_LAYERS = 4
FNO_PROJECT = 128
IN_CHANNELS = 2                      # (u, x)
BATCH_SIZE = 64

TIMED_STEPS = 30                     # timing only; not a training run
WARMUP_STEPS = 3


# ============================================================================
# Corpus
# ============================================================================
def build_steppers(dt: float):
    diffusion = ex.stepper.Diffusion(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt, diffusivity=DIFFUSIVITY
    )
    reaction = ex.stepper.reaction.FisherKPP(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt,
        diffusivity=REACTION_DIFFUSIVITY, reactivity=REACTIVITY, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION,
        num_circle_points=NUM_CIRCLE_POINTS, circle_radius=CIRCLE_RADIUS,
    )
    if float(np.asarray(reaction.diffusivity).reshape(-1)[0]) != REACTION_DIFFUSIVITY:
        raise SystemExit("component isolation failed: FisherKPP diffusivity is not zero")
    return diffusion, reaction


def initial_conditions(num_units: int, offset_range, seed: int) -> jnp.ndarray:
    base = ex.ic.RandomTruncatedFourierSeries(
        NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF, std_one=True
    )
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, num_units)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(
        offset_key, shape=(num_units, 1, 1),
        minval=offset_range[0], maxval=offset_range[1],
    )
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


def rollout(stepper, u0: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Return (units, steps+1, 1, NUM_POINTS)."""
    step = jax.jit(jax.vmap(stepper))
    frames = [u0]
    state = u0
    for _ in range(steps):
        state = step(state)
        frames.append(state)
    return jnp.stack(frames, axis=1)


# ============================================================================
# FNO-1d as a plain JAX pytree of real arrays. No optax, no equinox.
# Complex spectral weights are carried as separate real and imaginary arrays,
# mirroring the repository's existing torch decoder (view_as_complex).
# ============================================================================
def init_fno(key, width=FNO_WIDTH, modes=FNO_MODES, layers=FNO_LAYERS):
    keys = jr.split(key, 2 * layers + 4)
    scale = 1.0 / width
    params = {
        "lift_w": jr.normal(keys[0], (IN_CHANNELS, width)) * (1.0 / np.sqrt(IN_CHANNELS)),
        "lift_b": jnp.zeros((width,)),
        "spectral_re": [], "spectral_im": [], "local_w": [], "local_b": [],
        "proj1_w": jr.normal(keys[1], (width, FNO_PROJECT)) * (1.0 / np.sqrt(width)),
        "proj1_b": jnp.zeros((FNO_PROJECT,)),
        "proj2_w": jr.normal(keys[2], (FNO_PROJECT, 1)) * (1.0 / np.sqrt(FNO_PROJECT)),
        "proj2_b": jnp.zeros((1,)),
    }
    for i in range(layers):
        k1, k2 = jr.split(keys[3 + i])
        params["spectral_re"].append(jr.normal(k1, (width, width, modes)) * scale)
        params["spectral_im"].append(jr.normal(k2, (width, width, modes)) * scale)
        params["local_w"].append(jr.normal(k1, (width, width)) * (1.0 / np.sqrt(width)))
        params["local_b"].append(jnp.zeros((width,)))
    return params


def gelu(x):
    return jax.nn.gelu(x)


def spectral_conv(x, w_re, w_im, modes):
    """x: (batch, channels, N) -> (batch, channels, N)."""
    n = x.shape[-1]
    coeffs = jnp.fft.rfft(x, axis=-1)
    active = min(modes, coeffs.shape[-1])
    c = coeffs[..., :active]
    wr, wi = w_re[..., :active], w_im[..., :active]
    out_re = jnp.einsum("bim,iom->bom", c.real, wr) - jnp.einsum("bim,iom->bom", c.imag, wi)
    out_im = jnp.einsum("bim,iom->bom", c.real, wi) + jnp.einsum("bim,iom->bom", c.imag, wr)
    out = out_re + 1j * out_im
    padded = jnp.zeros(
        (x.shape[0], w_re.shape[1], coeffs.shape[-1]), dtype=out.dtype
    ).at[..., :active].set(out)
    return jnp.fft.irfft(padded, n=n, axis=-1)


def fno_apply(params, u, grid, modes=FNO_MODES):
    """u: (batch, 1, N) state. grid: (N,). Returns (batch, 1, N)."""
    batch, _, n = u.shape
    g = jnp.broadcast_to(grid, (batch, 1, n))
    x = jnp.concatenate([u, g], axis=1)                      # (b, 2, N)
    x = jnp.einsum("bcn,cw->bwn", x, params["lift_w"]) + params["lift_b"][None, :, None]
    for wr, wi, lw, lb in zip(
        params["spectral_re"], params["spectral_im"],
        params["local_w"], params["local_b"],
    ):
        update = spectral_conv(x, wr, wi, modes)
        update = update + jnp.einsum("bcn,co->bon", x, lw) + lb[None, :, None]
        x = x + gelu(update)
    x = gelu(jnp.einsum("bwn,wp->bpn", x, params["proj1_w"]) + params["proj1_b"][None, :, None])
    x = jnp.einsum("bpn,pk->bkn", x, params["proj2_w"]) + params["proj2_b"][None, :, None]
    return x


def loss_fn(params, u_in, u_out, grid):
    """Relative L2 per sample, averaged. Li et al.'s LpLoss."""
    pred = fno_apply(params, u_in, grid)
    num = jnp.sqrt(jnp.sum((pred - u_out) ** 2, axis=(1, 2)))
    den = jnp.sqrt(jnp.sum(u_out ** 2, axis=(1, 2)))
    return jnp.mean(num / den)


# ---------------------------------------------------------------- hand Adam
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (zeros, jax.tree_util.tree_map(jnp.zeros_like, params), jnp.array(0, jnp.int32))


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
    bc1 = 1 - b1 ** t.astype(jnp.float32)
    bc2 = 1 - b2 ** t.astype(jnp.float32)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * (a / bc1) / (jnp.sqrt(b / bc2) + eps), params, m, v
    )
    return params, (m, v, t)


@partial(jax.jit, static_argnums=())
def train_step(params, opt_state, u_in, u_out, grid, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, u_in, u_out, grid)
    params, opt_state = adam_update(params, grads, opt_state, lr)
    return params, opt_state, loss


# ============================================================================
def count_params(params):
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(params)))


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    record: dict = {
        "purpose": "wall-clock measurement only; no model was trained, no experiment run",
        "platform": {
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "jax": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
        },
        "importable": {},
        "config": {
            "num_points": NUM_POINTS, "frames_per_trajectory": FRAMES_PER_TRAJECTORY,
            "num_train_units": NUM_TRAIN_UNITS, "fno_modes": FNO_MODES,
            "fno_width": FNO_WIDTH, "fno_layers": FNO_LAYERS, "batch_size": BATCH_SIZE,
        },
    }
    for name in ("equinox", "optax", "torch", "flax", "jaxtyping", "scipy"):
        try:
            mod = __import__(name)
            record["importable"][name] = getattr(mod, "__version__", "present")
        except Exception as exc:
            record["importable"][name] = f"ABSENT ({type(exc).__name__})"

    diffusion, reaction = build_steppers(DT)
    grid64 = jnp.linspace(0.0, DOMAIN_EXTENT, NUM_POINTS, endpoint=False)

    # ------------------------------------------------ corpus: narrow diffusion
    u0_narrow = initial_conditions(NUM_TRAIN_UNITS, NARROW_OFFSET_RANGE, MASTER_SEED)
    traj, t_diff = timed(rollout, diffusion, u0_narrow, STEPS_PER_TRAJECTORY)
    record["corpus_seconds"] = {"diffusion_640x101": round(t_diff, 3)}
    record["corpus_shape"] = list(traj.shape)

    # ------------------------------------------------ corpus: reaction switch states
    u0_switch = initial_conditions(NUM_SWITCH_UNITS, NARROW_OFFSET_RANGE, MASTER_SEED + 1)
    switch_traj, t_reac = timed(rollout, reaction, u0_switch, STEPS_PER_TRAJECTORY)
    record["corpus_seconds"]["fisher_kpp_640x101"] = round(t_reac, 3)
    switch_states = switch_traj[:, -1]                      # (units, 1, N) at tau = 1.0

    # ------------------------------------------------ support statistics (data property)
    train_means = np.asarray(traj.mean(axis=(2, 3))).reshape(-1)     # every frame
    switch_means = np.asarray(switch_states.mean(axis=(1, 2)))
    record["support"] = {
        "note": "spatial means; a property of the corpus, not of any fitted model",
        "diffusion_training_frames_min": float(train_means.min()),
        "diffusion_training_frames_max": float(train_means.max()),
        "reaction_switch_min": float(switch_means.min()),
        "reaction_switch_max": float(switch_means.max()),
        "recorded_switch_range": list(RECORDED_SWITCH_MEAN_RANGE),
        "overlap": bool(
            switch_means.min() <= train_means.max()
            and train_means.min() <= switch_means.max()
        ),
        "gap": float(switch_means.min() - train_means.max()),
    }

    # ------------------------------------------------ observable resolution, data side
    # One diffusion step applied to switch states versus to training states: how big is
    # the signal the baseline must resolve, in the baseline's own relative-L2 units?
    step = jax.jit(jax.vmap(diffusion))
    onestep_switch = step(switch_states)
    onestep_train = step(traj[:, 0])
    def rel(a, b):
        num = np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(axis=(1, 2)))
        den = np.sqrt((np.asarray(b) ** 2).sum(axis=(1, 2)))
        return num / den
    record["signal"] = {
        "note": "relative L2 between input and one-step output; the change a one-step "
                "operator has to represent. Sets the scale a baseline error must beat.",
        "median_change_on_training_states": float(np.median(rel(onestep_train, traj[:, 0]))),
        "median_change_on_switch_states": float(np.median(rel(onestep_switch, switch_states))),
        "float32_eps_relative": float(np.finfo(np.float32).eps),
        "float64_eps_relative": float(np.finfo(np.float64).eps),
    }

    # ------------------------------------------------ training-step timing
    # Pairs: (u_t, u_{t+1}) for t in 0..99 over 640 trajectories.
    pairs_per_traj = STEPS_PER_TRAJECTORY
    record["num_training_pairs"] = NUM_TRAIN_UNITS * pairs_per_traj

    timings: dict = {}
    for label, dtype, width in (
        ("f32_width64", jnp.float32, 64),
        ("f32_width32", jnp.float32, 32),
        ("f64_width64", jnp.float64, 64),
    ):
        u_in = jnp.asarray(traj[:BATCH_SIZE, 0], dtype=dtype)
        u_out = jnp.asarray(traj[:BATCH_SIZE, 1], dtype=dtype)
        grid = jnp.asarray(grid64, dtype=dtype)
        params = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, dtype=dtype), init_fno(jr.PRNGKey(0), width=width)
        )
        opt_state = adam_init(params)
        lr = jnp.asarray(1e-3, dtype=dtype)

        t0 = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state, u_in, u_out, grid, lr)
        jax.block_until_ready(loss)
        compile_s = time.perf_counter() - t0

        for _ in range(WARMUP_STEPS):
            params, opt_state, loss = train_step(params, opt_state, u_in, u_out, grid, lr)
        jax.block_until_ready(loss)

        t0 = time.perf_counter()
        for _ in range(TIMED_STEPS):
            params, opt_state, loss = train_step(params, opt_state, u_in, u_out, grid, lr)
        jax.block_until_ready(loss)
        per_step = (time.perf_counter() - t0) / TIMED_STEPS

        timings[label] = {
            "params": count_params(params),
            "compile_seconds": round(compile_s, 3),
            "seconds_per_step": round(per_step, 5),
            "steps_per_second": round(1.0 / per_step, 2),
            "final_probe_loss": float(loss),
        }
    record["train_step_timing"] = timings

    # ------------------------------------------------ extrapolation
    ref = timings["f32_width64"]["seconds_per_step"]
    epochs = 100
    steps_per_epoch = record["num_training_pairs"] // BATCH_SIZE
    steps_budget = epochs * steps_per_epoch
    runs = 2 * 3          # 2 support conditions x 3 seeds
    record["extrapolation"] = {
        "seconds_per_step_f32_width64": ref,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "steps_per_run": steps_budget,
        "seconds_per_run": round(ref * steps_budget, 1),
        "runs": runs,
        "training_seconds_total": round(ref * steps_budget * runs, 1),
        "training_hours_total": round(ref * steps_budget * runs / 3600.0, 2),
        "corpus_seconds_total": round((t_diff + t_reac) * 2, 1),
    }

    out = HERE.parents[1] / "results" / "tables" / "baseline" / "TIMING_PROBE.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
