"""Family 8: Operator-algebra recovery on complex 2-D multi-operator dynamics,
plus external-data validation.

Governed by PREDECLARED.md (sha256 in results/PREDECLARED.sha256).

Subcommands:
    prescreen                   -> Truth-only screen for all 12 pairs and length-3 words,
                                   writes results/PRESCREEN.json
    smoke [steps]               -> Local CPU smoke at 32x32, 2 members (D, R), seed 0,
                                   one length-2 word, algebra search len<=2,
                                   writes pde_volume/family8/SMOKE.json
    train <member> <seed> <cond>-> Train FNO-2D for 8000 steps at 64x64 on member, seed,
                                   and condition ('narrow' or 'broadS').
                                   Saves ckpt, writes results/<cond>/<member>_seed<k>_pregate.json (e_on only)
    words                       -> Evaluate composed rollouts for all tested words and conditions,
                                   writes results/WORDS.json
    algebra                     -> Run exhaustive composition search for algebra recovery,
                                   writes results/ALGEBRA.json
    external                    -> Secondary external-data validation on PDEBench sample,
                                   writes results/EXTERNAL.json
    gates                       -> Check all 40 pregate files, verify G1, G2, G4a, P6,
                                   writes results/GATES.json
    evaluate                    -> Load checkpoints, evaluate P3, ALG-1, G4b, and test-time splitting,
                                   writes results/RESULT.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
import time
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import exponax as ex
from exponax._base_stepper import BaseNonlinearFun, BaseStepper

HERE = Path(__file__).resolve().parent

# Directories
WORK_DEFAULT = Path("pde_volume/family8")
WORK = Path(os.environ.get("FAMILY8_WORK_DIR", str(WORK_DEFAULT) if WORK_DEFAULT.parent.exists() else str(HERE / "work")))
RESULTS = HERE / "results"
CKPT = WORK / "ckpt"
LOGS = WORK / "logs"

# Problem parameters
NUM_SPATIAL_DIMS = 2
DOMAIN_EXTENT = 1.0
DT = 0.01
K = 10
FULL_GRID = 64
SMOKE_GRID = 32
SMOKE_STEPS = 30

DIFFUSIVITY = 0.01
REACTIVITY = 1.0
GRAY_SCOTT_F = 0.04
CELLULAR_SCALE = 1.0

FRAMES_PER_LEG = 101
STEPS_PER_LEG = 100

NUM_TRAIN_UNITS = 640
EVAL_UNITS = 32
HELDOUT_UNITS = 20
NARROW = (0.4865, 0.5171)
IC_AMPLITUDE = 0.175
IC_CUTOFF = 5
MASTER_SEED = 20260905

MEMBERS = ("D", "A", "R", "G")
SEEDS = (0, 1, 2, 3, 4)
# Predeclared reduction hook (PREDECLARED.md section 8): the seed set may be reduced
# from 5 to 3 on wall-time grounds. FAMILY8_SEEDS makes that reduction executable for
# the aggregation stages without editing the file mid-run.
_seeds_env = os.environ.get("FAMILY8_SEEDS", "").replace(",", " ").split()
if _seeds_env:
    SEEDS = tuple(int(x) for x in _seeds_env)
CONDITIONS = ("narrow", "broadS")

# Training hyperparameters
STEPS = 8000
BATCH = 64
LR_HIGH = 1e-3
LR_LOW = 1e-5

# Gate thresholds
E_ON_REF = 1.92247e-4
G2_MIN_DYNAMIC_RANGE = 100.0
G2_MAX_E_ON_FRACTION = 0.1
G4_MIN_E_ON = 1.19e-5
P6_MAX_E_ON = 1.573e-3
P3_MAX_BROAD_FRACTION = 0.25
ORDER_CONTRAST_FLOOR = 0.05
ALG1_MIN_BROAD_ACC = 0.80
ALG1_MIN_ACC_GAIN = 0.30

# Words list
PAIRS_ALL = tuple(f"{b}o{a}" for b in MEMBERS for a in MEMBERS if a != b)
LENGTH3_WORDS = ("RoAoD", "DoAoR", "GoDoA")


# ============================================================================
# Steppers: 2-D Physical Operators
# ============================================================================
class CellularAdvectionFun(BaseNonlinearFun):
    derivative_operator: jax.Array
    velocity: jax.Array
    scale: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        domain_extent: float,
        *,
        derivative_operator: jax.Array,
        scale: float = CELLULAR_SCALE,
        dealiasing_fraction: float = 2 / 3,
    ):
        super().__init__(num_spatial_dims, num_points, dealiasing_fraction=dealiasing_fraction)
        self.derivative_operator = derivative_operator
        self.scale = scale
        x = jnp.linspace(0.0, domain_extent, num_points, endpoint=False)
        y = jnp.linspace(0.0, domain_extent, num_points, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        k = 2.0 * jnp.pi / domain_extent
        vx = -jnp.sin(k * X) * jnp.cos(k * Y)
        vy = jnp.cos(k * X) * jnp.sin(k * Y)
        self.velocity = jnp.stack([vx, vy], axis=0)

    def __call__(self, u_hat: jax.Array) -> jax.Array:
        # u_hat: (1, H, W//2+1)
        grad_u_hat = self.derivative_operator * u_hat
        grad_u = self.ifft(grad_u_hat)
        v_dot_grad = jnp.sum(self.velocity * grad_u, axis=0, keepdims=True)
        return -self.scale * self.fft(v_dot_grad)


class CellularAdvectionStepper(BaseStepper):
    scale: float
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        scale: float = CELLULAR_SCALE,
        order: int = 4,
        dealiasing_fraction: float = 2 / 3,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        self.scale = scale
        self.dealiasing_fraction = dealiasing_fraction
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(self, derivative_operator):
        from exponax._spectral import wavenumber_shape
        shape = (1,) + wavenumber_shape(self.num_spatial_dims, self.num_points)
        return jnp.zeros(shape, dtype=jnp.complex64)

    def _build_nonlinear_fun(self, derivative_operator):
        return CellularAdvectionFun(
            self.num_spatial_dims,
            self.num_points,
            self.domain_extent,
            derivative_operator=derivative_operator,
            scale=self.scale,
            dealiasing_fraction=self.dealiasing_fraction,
        )


def build_stepper(member: str, num_points: int, dt: float = DT):
    """Factory for dictionary member steppers."""
    if member == "D":
        return ex.stepper.Diffusion(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, num_points, dt, diffusivity=DIFFUSIVITY
        )
    elif member == "A":
        return CellularAdvectionStepper(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, num_points, dt, scale=CELLULAR_SCALE, order=4
        )
    elif member == "R":
        return ex.stepper.reaction.FisherKPP(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, num_points, dt,
            diffusivity=0.0, reactivity=REACTIVITY, order=2,
            dealiasing_fraction=2 / 3, num_circle_points=16, circle_radius=1.0,
        )
    elif member == "G":
        F = GRAY_SCOTT_F
        return ex.stepper.generic.GeneralPolynomialStepper(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, num_points, dt,
            linear_coefficients=(-(1.0 + F), 0.0, 0.0),
            polynomial_coefficients=(F, 0.0, 2.0, -1.0),
            order=2, dealiasing_fraction=2 / 3,
        )
    elif member == "AC":
        return ex.stepper.reaction.AllenCahn(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, num_points, dt,
            diffusivity=0.005, first_order_coefficient=1.0, third_order_coefficient=-1.0,
            order=2, dealiasing_fraction=0.5,
        )
    else:
        raise ValueError(f"Unknown member {member!r}")


# ============================================================================
# Initial Conditions, Rollouts, and Corpora
# ============================================================================
def initial_conditions_2d(
    num_units: int, offset_range: tuple[float, float], seed: int, num_points: int
) -> jnp.ndarray:
    base = ex.ic.RandomTruncatedFourierSeries(
        NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF, std_one=True
    )
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, num_units)
    raw = jnp.stack([base(num_points, key=k) for k in keys])  # (num_units, 1, H, W)
    offsets = jr.uniform(
        offset_key, shape=(num_units, 1, 1, 1),
        minval=offset_range[0], maxval=offset_range[1],
    )
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


@eqx.filter_jit
def _stepper_apply(stepper, u: jnp.ndarray) -> jnp.ndarray:
    """One vmapped truth step. Compiled once per (stepper structure, batch shape)."""
    return jax.vmap(stepper)(u)


def rollout_2d(stepper, u0: jnp.ndarray, steps: int) -> jnp.ndarray:
    """Return (units, steps+1, 1, H, W)."""
    frames = [u0]
    state = u0
    for _ in range(steps):
        state = _stepper_apply(stepper, state)
        frames.append(state)
    return jnp.stack(frames, axis=1)


def make_grid_2d(num_points: int) -> jnp.ndarray:
    gx = jnp.linspace(0.0, DOMAIN_EXTENT, num_points, endpoint=False)
    gy = jnp.linspace(0.0, DOMAIN_EXTENT, num_points, endpoint=False)
    grid_x, grid_y = jnp.meshgrid(gx, gy, indexing="ij")
    return jnp.stack([grid_x, grid_y], axis=0)


def make_pairs_2d(states: jnp.ndarray, k: int = K) -> tuple[jnp.ndarray, jnp.ndarray]:
    h, w = states.shape[-2], states.shape[-1]
    pairs_in = states[:, :-k].reshape(-1, 1, h, w)
    pairs_out = states[:, k:].reshape(-1, 1, h, w)
    return pairs_in, pairs_out


def spatial_means_2d(states: jnp.ndarray | np.ndarray) -> np.ndarray:
    return np.asarray(states.mean(axis=(-2, -1))).reshape(-1)


def relative_l2_2d(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    num = np.sqrt(((pred - target) ** 2).sum(axis=(1, 2, 3)))
    den = np.sqrt((target ** 2).sum(axis=(1, 2, 3)))
    return num / den


def parse_word(word_str: str) -> list[str]:
    """Parse word string like 'RoAoD' into ['D', 'A', 'R'] (in order of application: right to left)."""
    parts = word_str.split("o")
    # Mathematical notation w = O_k o ... o O_1 means O_1 is applied first, then O_2, etc.
    return list(reversed(parts))


def rollout_word_truth(word_ops: list[str], u0: jnp.ndarray, num_points: int, steps_per_leg: int = STEPS_PER_LEG) -> list[jnp.ndarray]:
    """Execute truth multi-leg trajectory. Returns list of leg trajectory arrays, each (units, steps+1, 1, H, W)."""
    curr = u0
    legs = []
    for member in word_ops:
        op = build_stepper(member, num_points)
        traj = rollout_2d(op, curr, steps_per_leg)
        legs.append(traj)
        curr = traj[:, -1]
    return legs


def build_broad_s_union_corpus(
    words_list: list[str], num_points: int = FULL_GRID, n_units: int = NUM_TRAIN_UNITS
) -> jnp.ndarray:
    """Build union of switch states across words_list, sampled to exactly n_units."""
    base_ics = initial_conditions_2d(n_units, NARROW, MASTER_SEED + 1, num_points)
    n_base = n_units // 4  # 160 units (25% baseline support as in family 1e/1c)
    n_switch_needed = n_units - n_base  # 480 units

    # Harvest intermediate switch states from each word in words_list
    switch_pool = []
    units_per_word = max(4, n_switch_needed // len(words_list))
    for w_idx, w_str in enumerate(words_list):
        word_ops = parse_word(w_str)
        sub_ic = base_ics[w_idx * units_per_word : (w_idx + 1) * units_per_word]
        if sub_ic.shape[0] == 0:
            sub_ic = base_ics[:units_per_word]
        legs = rollout_word_truth(word_ops, sub_ic, num_points)
        # Collect switch states at the handoff between legs
        for leg in legs[:-1]:
            switch_pool.append(leg[:, -1])  # endpoint of leg is switch state
        # Also intermediate burst states at 50% leg
        for leg in legs:
            switch_pool.append(leg[:, STEPS_PER_LEG // 2])

    switch_all = jnp.concatenate(switch_pool, axis=0)
    # Subsample exactly n_switch_needed, without replacement whenever the pool allows
    # it (the pool is ~1700 states for the 10-word list, so it always does here).
    replace = bool(switch_all.shape[0] < n_switch_needed)
    idx = jr.choice(
        jr.PRNGKey(MASTER_SEED + 101), switch_all.shape[0],
        shape=(n_switch_needed,), replace=replace,
    )
    switch_sampled = switch_all[idx]

    broad_s_ics = jnp.concatenate([base_ics[:n_base], switch_sampled], axis=0)
    return broad_s_ics


def broad_s_corpus_cached(words_list: list[str], num_points: int = FULL_GRID,
                          n_units: int = NUM_TRAIN_UNITS) -> jnp.ndarray:
    """build_broad_s_union_corpus with an on-disk cache.

    The corpus is a deterministic function of the word list, the grid and n, and is
    identical for every member and seed, so it is built once per run instead of once
    per fit (20 rebuilds of ~2500 truth steps each)."""
    tag = "_".join(sorted(words_list))
    # hashlib, not hash(): Python randomises string hashing per process, so hash(tag)
    # produced a different cache key in every fit and the corpus was rebuilt each time.
    key = ("broadS_corpus_n{}_g{}_{}.npy".format(
        n_units, num_points, hashlib.sha256(tag.encode()).hexdigest()[:12]))
    path = WORK / "corpus" / key
    if path.exists():
        return jnp.asarray(np.load(path))
    ic = build_broad_s_union_corpus(words_list, num_points=num_points, n_units=n_units)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, np.asarray(ic))
    tmp.rename(path)
    (path.with_suffix(".words.json")).write_text(json.dumps(sorted(words_list), indent=2) + "\n")
    return ic


def train_states_2d(
    member: str, condition: str, tested_words: list[str] | None = None, num_points: int = FULL_GRID
) -> jnp.ndarray:
    """Return (640, 101, 1, H, W) training states."""
    op = build_stepper(member, num_points)
    if condition == "narrow":
        ic = initial_conditions_2d(NUM_TRAIN_UNITS, NARROW, MASTER_SEED + 1, num_points)
    elif condition == "broadS":
        if tested_words is None:
            tested_words = list(LENGTH3_WORDS)
        ic = broad_s_corpus_cached(tested_words, num_points=num_points, n_units=NUM_TRAIN_UNITS)
    else:
        raise ValueError(f"Unknown condition {condition!r}")
    return rollout_2d(op, ic, STEPS_PER_LEG)


def eval_cohorts_2d(member: str, num_points: int = FULL_GRID) -> dict[str, jnp.ndarray]:
    """On-support cohort for ONE dictionary member.

    Family 1e's construction (held-out narrow ICs at frame 0, target K steps later),
    except that the estimand is the member's own K-step flow map: in family 8 every
    member has its own surrogate, so e_on(member) must be measured against that
    member's operator, not against diffusion."""
    op = build_stepper(member, num_points)
    on_states = initial_conditions_2d(EVAL_UNITS, NARROW, MASTER_SEED + 7, num_points)
    on_target = rollout_2d(op, on_states, K)[:, -1]
    return {"on_states": on_states, "on_target": on_target}


# ============================================================================
# FNO-2D Model (Equinox)
# ============================================================================
class SpectralConv2d(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)
    w1_re: jax.Array
    w1_im: jax.Array
    w2_re: jax.Array
    w2_im: jax.Array

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        k1, k2, k3, k4 = jr.split(key, 4)
        scale = 1.0 / (in_channels * out_channels)
        self.w1_re = jr.normal(k1, (in_channels, out_channels, modes1, modes2)) * scale
        self.w1_im = jr.normal(k2, (in_channels, out_channels, modes1, modes2)) * scale
        self.w2_re = jr.normal(k3, (in_channels, out_channels, modes1, modes2)) * scale
        self.w2_im = jr.normal(k4, (in_channels, out_channels, modes1, modes2)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        c_in, h, w = x.shape
        coeffs = jnp.fft.rfft2(x, axes=(-2, -1))
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, coeffs.shape[-1])

        c1 = coeffs[:, :m1, :m2]
        w1_re = self.w1_re[:, :, :m1, :m2]
        w1_im = self.w1_im[:, :, :m1, :m2]
        out1_re = jnp.einsum("ixy,ioxy->oxy", c1.real, w1_re) - jnp.einsum("ixy,ioxy->oxy", c1.imag, w1_im)
        out1_im = jnp.einsum("ixy,ioxy->oxy", c1.real, w1_im) + jnp.einsum("ixy,ioxy->oxy", c1.imag, w1_re)
        out1 = out1_re + 1j * out1_im

        c2 = coeffs[:, -m1:, :m2]
        w2_re = self.w2_re[:, :, :m1, :m2]
        w2_im = self.w2_im[:, :, :m1, :m2]
        out2_re = jnp.einsum("ixy,ioxy->oxy", c2.real, w2_re) - jnp.einsum("ixy,ioxy->oxy", c2.imag, w2_im)
        out2_im = jnp.einsum("ixy,ioxy->oxy", c2.real, w2_im) + jnp.einsum("ixy,ioxy->oxy", c2.imag, w2_re)
        out2 = out2_re + 1j * out2_im

        out_ft = jnp.zeros((self.out_channels, h, coeffs.shape[-1]), dtype=coeffs.dtype)
        out_ft = out_ft.at[:, :m1, :m2].set(out1)
        out_ft = out_ft.at[:, -m1:, :m2].set(out2)
        return jnp.fft.irfft2(out_ft, s=(h, w), axes=(-2, -1))


class FNOBlock2d(eqx.Module):
    spectral: SpectralConv2d
    local: eqx.nn.Conv2d

    def __init__(self, width: int, modes1: int, modes2: int, *, key):
        k1, k2 = jr.split(key)
        self.spectral = SpectralConv2d(width, width, modes1, modes2, key=k1)
        self.local = eqx.nn.Conv2d(width, width, kernel_size=1, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + jax.nn.gelu(self.spectral(x) + self.local(x))


class FNO2d(eqx.Module):
    lift: eqx.nn.Conv2d
    blocks: tuple[FNOBlock2d, ...]
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        layers: int = 4,
        project_dim: int = 64,
        *,
        key,
    ):
        keys = iter(jr.split(key, layers + 3))
        self.lift = eqx.nn.Conv2d(in_channels, width, kernel_size=1, key=next(keys))
        self.blocks = tuple(FNOBlock2d(width, modes1, modes2, key=next(keys)) for _ in range(layers))
        self.proj1 = eqx.nn.Conv2d(width, project_dim, kernel_size=1, key=next(keys))
        self.proj2 = eqx.nn.Conv2d(project_dim, out_channels, kernel_size=1, key=next(keys))

    def __call__(self, u: jax.Array, grid: jax.Array) -> jax.Array:
        x = jnp.concatenate([u, grid], axis=0)
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        x = jax.nn.gelu(self.proj1(x))
        return self.proj2(x)


@eqx.filter_jit
def _model_apply(model, u: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """One vmapped surrogate step. Compiled once per batch shape, reused for every
    member/seed/condition because the weights enter as a traced argument."""
    return jax.vmap(lambda x: model(x, grid))(u)


def count_parameters(model: eqx.Module) -> int:
    return sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def save_checkpoint(path: Path, model: eqx.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves, _ = jax.tree_util.tree_flatten(model)
    np.savez(path, **{f"arr_{i}": np.asarray(x) for i, x in enumerate(leaves) if hasattr(x, "shape")})


def load_checkpoint(path: Path, template: eqx.Module) -> eqx.Module:
    leaves, treedef = jax.tree_util.tree_flatten(template)
    with np.load(path) as data:
        arrays = [jnp.asarray(data[f"arr_{i}"]) if hasattr(leaf, "shape") else leaf for i, leaf in enumerate(leaves)]
    return jax.tree_util.tree_unflatten(treedef, arrays)


# ============================================================================
# Optimizer and Training
# ============================================================================
def adam_init(model: eqx.Module):
    return (
        jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else None, model),
        jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_array(x) else None, model),
        jnp.array(0, jnp.int32),
    )


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1.0 - b1) * g if eqx.is_array(a) else None, m, grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1.0 - b2) * g * g if eqx.is_array(a) else None, v, grads)
    bc1 = 1.0 - b1 ** t.astype(jnp.float32)
    bc2 = 1.0 - b2 ** t.astype(jnp.float32)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * (a / bc1) / (jnp.sqrt(b / bc2) + eps) if eqx.is_array(p) else p,
        params, m, v,
    )
    return params, (m, v, t)


def loss_fn(model: FNO2d, u_in: jax.Array, u_out: jax.Array, grid: jax.Array) -> jax.Array:
    pred = jax.vmap(lambda x: model(x, grid))(u_in)
    num = jnp.sqrt(jnp.sum((pred - u_out) ** 2, axis=(1, 2, 3)))
    den = jnp.sqrt(jnp.sum(u_out ** 2, axis=(1, 2, 3)))
    return jnp.mean(num / den)


def cosine_lr(step: int, total_steps: int = STEPS) -> jnp.ndarray:
    lr = LR_LOW + 0.5 * (LR_HIGH - LR_LOW) * (1.0 + np.cos(np.pi * step / total_steps))
    return jnp.asarray(lr, jnp.float32)


@eqx.filter_jit
def train_step(model: FNO2d, opt_state, u_in: jax.Array, u_out: jax.Array, grid: jax.Array, lr: jax.Array):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, u_in, u_out, grid)
    model, opt_state = adam_update(model, grads, opt_state, lr)
    return model, opt_state, loss


def evaluate_model(model: FNO2d, states_in: jnp.ndarray, targets: jnp.ndarray, grid: jnp.ndarray) -> np.ndarray:
    preds = []
    for start in range(0, states_in.shape[0], BATCH):
        chunk = states_in[start : start + BATCH]
        pred_chunk = _model_apply(model, chunk, grid)
        preds.append(np.asarray(pred_chunk, dtype=np.float64))
    pred_all = np.concatenate(preds, axis=0)
    return relative_l2_2d(pred_all, np.asarray(targets, dtype=np.float64))


# ============================================================================
# SUBCOMMAND: prescreen
# ============================================================================
def prescreen(num_points: int = FULL_GRID) -> dict:
    """Truth-only pre-screen on all 12 pairs and length-3 words."""
    started = time.perf_counter()
    print(f"=== FAMILY 8 PRE-SCREEN: {num_points}x{num_points} truth dynamics ===", flush=True)

    n_ps = 32
    base_ic = initial_conditions_2d(n_ps, NARROW, MASTER_SEED, num_points)
    steppers = {m: build_stepper(m, num_points) for m in MEMBERS}

    # 1. Single leg supports
    single_supports = {}
    for m in MEMBERS:
        traj = rollout_2d(steppers[m], base_ic, STEPS_PER_LEG)
        means = spatial_means_2d(traj[:, -1])
        single_supports[m] = {"min": float(means.min()), "max": float(means.max()), "mean": float(means.mean())}
        print(f"  {m} single leg endpoint support: [{single_supports[m]['min']:.4f}, {single_supports[m]['max']:.4f}]", flush=True)

    # 2. Pairs pre-screen
    pairs_results = {}
    passing_pairs = []
    for b in MEMBERS:
        for a in MEMBERS:
            if a == b:
                continue
            pair_name = f"{b}o{a}"
            # A runs first (100 steps)
            u_sw = rollout_2d(steppers[a], base_ic, STEPS_PER_LEG)[:, -1]
            sw_means = spatial_means_2d(u_sw)
            sw_min, sw_max = float(sw_means.min()), float(sw_means.max())

            # Support shift vs receiving operator B
            b_sup = (single_supports[b]["min"], single_supports[b]["max"])
            overlap = max(0.0, min(b_sup[1], sw_max) - max(b_sup[0], sw_min))
            gap = max(sw_min - b_sup[1], b_sup[0] - sw_max)
            g1_spatial_pass = bool(overlap == 0.0 and gap > 0.0)

            # PCA-99 Mahalanobis support shift
            u_b_nrw = rollout_2d(steppers[b], base_ic, STEPS_PER_LEG)[:, -1]
            x_b = np.asarray(u_b_nrw, np.float64).reshape(n_ps, -1)
            x_sw = np.asarray(u_sw, np.float64).reshape(n_ps, -1)
            mu = x_b.mean(axis=0)
            xc = x_b - mu
            cov = (xc.T @ xc) / (n_ps - 1)
            lam, vec = np.linalg.eigh(cov)
            order = np.argsort(lam)[::-1]
            lam, vec = lam[order], vec[:, order]
            cum = np.cumsum(lam) / max(lam.sum(), 1e-12)
            k_pca = int(np.searchsorted(cum, 0.99) + 1)
            u_pca = vec[:, :k_pca]
            lam_k = np.maximum(lam[:k_pca], 1e-12)
            md_tr = np.sqrt(((xc @ u_pca) ** 2 / lam_k).sum(axis=1))
            r99 = float(np.percentile(md_tr, 99))
            md_sw = np.sqrt((((x_sw - mu) @ u_pca) ** 2 / lam_k).sum(axis=1))
            pca_coverage = float((md_sw <= r99).mean())
            g1_pca_pass = bool(pca_coverage == 0.0)
            g1_pass = g1_spatial_pass or g1_pca_pass

            # Dynamic range G2
            target_b = rollout_2d(steppers[b], u_sw, K)[:, -1]
            pers_off = float(np.median(relative_l2_2d(np.asarray(u_sw, np.float64), np.asarray(target_b, np.float64))))
            dr = pers_off / E_ON_REF
            g2_pass = bool(dr >= G2_MIN_DYNAMIC_RANGE)

            # Order contrast
            u_ba = rollout_2d(steppers[b], u_sw, STEPS_PER_LEG)[:, -1]
            u_ab = rollout_2d(steppers[a], rollout_2d(steppers[b], base_ic, STEPS_PER_LEG)[:, -1], STEPS_PER_LEG)[:, -1]
            diff = np.asarray(jnp.abs(u_ba - u_ab), np.float64)
            contrast_rel = float(np.median(np.sqrt((diff ** 2).sum(axis=(1, 2, 3))) / np.sqrt((np.asarray(u_ba, np.float64) ** 2).sum(axis=(1, 2, 3)))))
            contrast_max = float(diff.max())
            order_pass = bool(contrast_rel >= ORDER_CONTRAST_FLOOR or contrast_max >= ORDER_CONTRAST_FLOOR)

            pair_record = {
                "pair": pair_name,
                "first": a, "second": b,
                "gap": gap, "overlap": overlap, "g1_spatial_pass": g1_spatial_pass,
                "pca_k": k_pca, "pca_coverage": pca_coverage, "g1_pca_pass": g1_pca_pass, "g1_pass": g1_pass,
                "persistence_off": pers_off, "proxy_dynamic_range": dr, "g2_pass": g2_pass,
                "contrast_relative_l2": contrast_rel, "contrast_max_abs": contrast_max, "order_pass": order_pass,
                "survives": bool(g1_pass and g2_pass and order_pass),
            }
            pairs_results[pair_name] = pair_record
            if pair_record["survives"]:
                passing_pairs.append(pair_name)
            print(f"  {pair_name:6s} | gap={gap:+.4f} | DR={dr:7.1f} (G2={g2_pass}) | "
                  f"contrast={contrast_rel:.4f} (Order={order_pass}) | survives={pair_record['survives']}", flush=True)

    # 3. Length 3 words
    len3_results = {}
    for w_str in LENGTH3_WORDS:
        word_ops = parse_word(w_str)
        legs = rollout_word_truth(word_ops, base_ic, num_points)
        end_means = spatial_means_2d(legs[-1][:, -1])
        len3_results[w_str] = {
            "endpoint_mean_support": [float(end_means.min()), float(end_means.max())],
            "survives": True,
        }
        print(f"  {w_str:8s} endpoint support: [{float(end_means.min()):.4f}, {float(end_means.max()):.4f}]", flush=True)

    elapsed = time.perf_counter() - started
    payload = {
        "schema": "family8-prescreen-v1",
        "num_points": num_points,
        "single_leg_supports": single_supports,
        "pairs": pairs_results,
        "passing_pairs": passing_pairs,
        "length3_words": len3_results,
        "elapsed_seconds": elapsed,
        "status": "PASS: Pre-screen executed",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "PRESCREEN.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote PRESCREEN.json to {RESULTS / 'PRESCREEN.json'} ({elapsed:.1f}s)", flush=True)
    return payload


# ============================================================================
# SUBCOMMAND: smoke
# ============================================================================
def smoke(steps: int = SMOKE_STEPS) -> None:
    """CPU smoke test at 32x32, 2 members (D, R), seed 0, 30 steps, one word, algebra search len<=2."""
    started = time.perf_counter()
    print(f"=== FAMILY 8 CPU SMOKE: {SMOKE_GRID}x{SMOKE_GRID}, 2 members (D, R), {steps} steps, seed 0 ===", flush=True)

    grid = make_grid_2d(SMOKE_GRID)
    members_smoke = ["D", "R"]

    # 1. Prepare data for D and R
    print("Generating smoke training data for D and R...", flush=True)
    t0 = time.perf_counter()
    train_data = {}
    for m in members_smoke:
        op = build_stepper(m, SMOKE_GRID)
        ic = initial_conditions_2d(64, NARROW, MASTER_SEED + 1, SMOKE_GRID)
        states = rollout_2d(op, ic, STEPS_PER_LEG)
        p_in, p_out = make_pairs_2d(states)
        train_data[m] = (p_in, p_out)
    print(f"  Data generated in {time.perf_counter() - t0:.2f}s", flush=True)

    # 2. Train models for D and R (30 steps each)
    models = {}
    pregate_e_on = {}

    for m in members_smoke:
        eval_data = eval_cohorts_2d(m, SMOKE_GRID)
        print(f"Training smoke surrogate for {m} ({steps} steps)...", flush=True)
        model = FNO2d(key=jr.PRNGKey(0))
        opt_state = adam_init(model)
        p_in, p_out = train_data[m]
        n_pairs = p_in.shape[0]

        for s in range(steps):
            idx = jr.randint(jr.PRNGKey(s + 1000), (BATCH,), 0, n_pairs)
            lr = cosine_lr(s, steps)
            model, opt_state, loss = train_step(model, opt_state, p_in[idx], p_out[idx], grid, lr)

        models[m] = model
        e_on_arr = evaluate_model(model, eval_data["on_states"], eval_data["on_target"], grid)
        e_on = float(np.median(e_on_arr))
        pregate_e_on[m] = e_on
        print(f"  {m} e_on: {e_on:.6e}", flush=True)

    # 3. Checkpoint save & reload
    smoke_ckpt = WORK / "ckpt" / "smoke_D_seed0.npz"
    save_checkpoint(smoke_ckpt, models["D"])
    reloaded_D = load_checkpoint(smoke_ckpt, FNO2d(key=jr.PRNGKey(999)))
    eval_D = eval_cohorts_2d("D", SMOKE_GRID)
    e_on_reload = float(np.median(evaluate_model(reloaded_D, eval_D["on_states"], eval_D["on_target"], grid)))
    reload_exact = bool(np.isclose(e_on_reload, pregate_e_on["D"], atol=1e-6))
    print(f"  Checkpoint reload verified exact: {reload_exact}", flush=True)

    # 4. Rollout composed word: R o D (D runs first, then R)
    print("Testing composed word rollout: RoD...", flush=True)
    test_ic = initial_conditions_2d(8, NARROW, MASTER_SEED + 77, SMOKE_GRID)
    stepper_D = build_stepper("D", SMOKE_GRID)
    stepper_R = build_stepper("R", SMOKE_GRID)

    truth_leg1 = rollout_2d(stepper_D, test_ic, STEPS_PER_LEG)
    truth_leg2 = rollout_2d(stepper_R, truth_leg1[:, -1], STEPS_PER_LEG)
    # Stride K truth frames: (units, 21, 1, H, W)
    truth_frames = jnp.concatenate([truth_leg1[:, ::K], truth_leg2[:, K::K]], axis=1)

    # Surrogate rollout: apply models['D'] for 10 steps, then models['R'] for 10 steps
    surr_frames = [test_ic]
    curr = test_ic
    for _ in range(10):
        curr = _model_apply(models["D"], curr, grid)
        surr_frames.append(curr)
    for _ in range(10):
        curr = _model_apply(models["R"], curr, grid)
        surr_frames.append(curr)
    surr_stacked = jnp.stack(surr_frames, axis=1)
    word_err = float(np.median(relative_l2_2d(np.asarray(surr_stacked[:, -1], np.float64), np.asarray(truth_frames[:, -1], np.float64))))
    print(f"  Composed word RoD endpoint error: {word_err:.6e}", flush=True)

    # 5. Algebra Recovery search over words of length <= 2
    print("Running smoke algebra recovery search (length <= 2)...", flush=True)
    # Candidates: 'D', 'R', 'DoD', 'DoR', 'RoD', 'RoR'
    candidates = ["D", "R", "DoD", "DoR", "RoD", "RoR"]
    cand_scores = {}
    for cand in candidates:
        ops = parse_word(cand)
        curr = test_ic
        cand_traj = [curr]
        for op_m in ops:
            for _ in range(10):
                curr = _model_apply(models[op_m], curr, grid)
                cand_traj.append(curr)
        cand_stacked = jnp.stack(cand_traj, axis=1)
        # Compare on common frames
        min_len = min(cand_stacked.shape[1], truth_frames.shape[1])
        diff = np.asarray(cand_stacked[:, :min_len] - truth_frames[:, :min_len], np.float64)
        den = np.asarray(truth_frames[:, :min_len], np.float64)
        score = float(np.mean(np.sqrt((diff ** 2).sum(axis=(-2, -1))) / np.sqrt((den ** 2).sum(axis=(-2, -1)))))
        if cand_stacked.shape[1] < truth_frames.shape[1]:
            score += 0.5  # Length penalty
        cand_scores[cand] = score

    best_word = min(cand_scores, key=cand_scores.get)
    print(f"  Candidate scores: {cand_scores}")
    print(f"  Top-1 recovered word: {best_word} (Truth: RoD)", flush=True)

    elapsed = time.perf_counter() - started
    smoke_record = {
        "schema": "family8-smoke-v1",
        "grid": [SMOKE_GRID, SMOKE_GRID],
        "members": members_smoke,
        "seed": 0,
        "steps": steps,
        "fno_params": count_parameters(models["D"]),
        "pregate_e_on": pregate_e_on,
        "checkpoint_reload_exact": reload_exact,
        "composed_word": "RoD",
        "composed_endpoint_error": word_err,
        "algebra_scores": cand_scores,
        "top1_word": best_word,
        "truth_word": "RoD",
        "elapsed_seconds": elapsed,
        "platform": {
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "jax": jax.__version__,
        },
        "status": "PASS: 2-D smoke fit, checkpoint save/reload, composed rollout, and algebra search verified",
    }

    smoke_dest = WORK / "SMOKE.json"
    smoke_dest.parent.mkdir(parents=True, exist_ok=True)
    smoke_dest.write_text(json.dumps(smoke_record, indent=2) + "\n")
    print(f"\nWrote SMOKE.json to {smoke_dest} in {elapsed:.1f}s total.", flush=True)


# ============================================================================
# SUBCOMMAND: train <member> <seed> <cond>
# ============================================================================
def train(member: str, seed: int, condition: str) -> None:
    """Train FNO-2D for 8000 steps at 64x64 on member, seed, and condition."""
    if member not in MEMBERS:
        raise ValueError(f"Unknown member {member!r} not in {MEMBERS}")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed {seed} not in {SEEDS}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r} not in {CONDITIONS}")

    started = time.perf_counter()
    print(f"=== TRAINING Family 8: member={member}, seed={seed}, condition={condition}, grid={FULL_GRID}x{FULL_GRID} ===", flush=True)

    grid = make_grid_2d(FULL_GRID)
    # Get passing words from PRESCREEN if available, else default
    ps_file = RESULTS / "PRESCREEN.json"
    tested_words = list(LENGTH3_WORDS)
    if ps_file.exists():
        ps_data = json.loads(ps_file.read_text())
        tested_words = ps_data.get("passing_pairs", []) + list(LENGTH3_WORDS)

    ts = train_states_2d(member, condition, tested_words=tested_words, num_points=FULL_GRID)
    pairs_in, pairs_out = make_pairs_2d(ts)
    del ts

    model = FNO2d(key=jr.PRNGKey(seed))
    opt_state = adam_init(model)
    key = jr.PRNGKey(seed + 9999)
    n_pairs = pairs_in.shape[0]

    t0 = time.perf_counter()
    loss_trace = {}
    for step in range(STEPS):
        key, batch_key = jr.split(key)
        idx = jr.randint(batch_key, (BATCH,), 0, n_pairs)
        lr = cosine_lr(step, STEPS)
        model, opt_state, loss = train_step(model, opt_state, pairs_in[idx], pairs_out[idx], grid, lr)
        if step == 0 or (step + 1) % 1000 == 0:
            loss_trace[str(step + 1)] = float(loss)
            print(f"  step {step+1}/{STEPS}  train loss {float(loss):.6e}", flush=True)
    train_seconds = time.perf_counter() - t0
    del pairs_in, pairs_out, opt_state

    # Evaluate e_on ONLY
    eval_data = eval_cohorts_2d(member, num_points=FULL_GRID)
    e_on_arr = evaluate_model(model, eval_data["on_states"], eval_data["on_target"], grid)
    e_on = float(np.median(e_on_arr))

    # Save checkpoint
    ckpt_path = CKPT / f"{condition}_{member}_seed{seed}.npz"
    save_checkpoint(ckpt_path, model)

    # Save pregate file
    cond_dir = RESULTS / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    pregate_payload = {
        "schema": "family8-pregate-v1",
        "member": member,
        "seed": seed,
        "condition": condition,
        "grid": [FULL_GRID, FULL_GRID],
        "steps": STEPS,
        "batch": BATCH,
        "e_on": e_on,
        "train_loss_trace": loss_trace,
        "train_seconds": train_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(ckpt_path),
        "note": "e_off is strictly omitted; gates must be cleared first",
        "platform": {
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "jax": jax.__version__,
        },
    }
    (cond_dir / f"{member}_seed{seed}_pregate.json").write_text(json.dumps(pregate_payload, indent=2) + "\n")
    print(f"Saved pregate: {member} {condition} seed {seed} e_on={e_on:.6e} ({train_seconds:.1f}s)", flush=True)


# ============================================================================
# SUBCOMMAND: words
# ============================================================================
def tested_word_list() -> list[str]:
    """Words under test: the pre-screen survivors plus the predeclared length-3 words."""
    ps_file = RESULTS / "PRESCREEN.json"
    if not ps_file.exists():
        raise SystemExit("Missing PRESCREEN.json. Run prescreen first.")
    ps_data = json.loads(ps_file.read_text())
    return list(ps_data.get("passing_pairs", [])) + list(LENGTH3_WORDS)


def truth_trajectory_for(word_ops: list[str], u0: jnp.ndarray, num_points: int = FULL_GRID) -> jnp.ndarray:
    """Truth trajectory of a word on the stride-K estimand grid: (units, 1 + legs*K_steps, 1, H, W)."""
    legs = rollout_word_truth(word_ops, u0, num_points)
    frames = [legs[0][:, 0]]
    for leg in legs:
        frames.extend([leg[:, idx] for idx in range(K, STEPS_PER_LEG + 1, K)])
    return jnp.stack(frames, axis=1)


def surrogate_rollout(seq: list[str], models: dict, u0: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Autoregressive composed surrogate rollout. `seq` lists one member per application."""
    curr = u0
    frames = [curr]
    for m in seq:
        curr = _model_apply(models[m], curr, grid)
        frames.append(curr)
    return jnp.stack(frames, axis=1)


def word_sequence(word_ops: list[str], n_leg: int = STEPS_PER_LEG // K) -> list[str]:
    return [m for m in word_ops for _ in range(n_leg)]


def splitting_schemes(word_ops: list[str], n_leg: int = STEPS_PER_LEG // K) -> dict[str, list[str]]:
    """Test-time operator-splitting schemes for one word (family 6 METHOD_SPEC.md).

    Every scheme spends the same number of surrogate applications, advances the same
    total time, uses the same narrow-trained dictionary and adds no training; only the
    test-time ordering differs. `leg_sequential_lie` is the word order itself, so the
    granted test-time search contains the correct ordering as one of its candidates.
    """
    fwd = list(word_ops)
    schemes = {
        "leg_sequential_lie": word_sequence(word_ops, n_leg),
        "lie_fine": fwd * n_leg,
    }
    if n_leg % 2 == 0:
        schemes["strang_fine"] = (fwd + fwd[::-1]) * (n_leg // 2)
    return schemes


def traj_rel_l2_frames(pred: jnp.ndarray, target: jnp.ndarray) -> np.ndarray:
    """Per-unit mean over frames 1..T-1 of the relative L2 error (predeclared score)."""
    p, y = pred[:, 1:], target[:, 1:]
    num = jnp.sqrt(((p - y) ** 2).sum(axis=(-3, -2, -1)))
    den = jnp.sqrt((y ** 2).sum(axis=(-3, -2, -1)))
    return np.asarray((num / den).mean(axis=1), np.float64)


def load_dictionary(cond: str, seed: int) -> dict:
    models = {}
    for m in MEMBERS:
        ckpt_p = CKPT / f"{cond}_{m}_seed{seed}.npz"
        if not ckpt_p.exists():
            raise SystemExit(f"Missing checkpoint: {ckpt_p}")
        models[m] = load_checkpoint(ckpt_p, FNO2d(key=jr.PRNGKey(seed)))
    return models



def words() -> None:
    """Evaluate composed rollouts for all tested words across conditions and seeds."""
    started = time.perf_counter()
    print("=== EVALUATING COMPOSED ROLLOUTS (WORDS) ===", flush=True)

    grid = make_grid_2d(FULL_GRID)
    tested_words = tested_word_list()
    test_ic = initial_conditions_2d(EVAL_UNITS, NARROW, MASTER_SEED + 88, FULL_GRID)

    dicts = {c: {s: load_dictionary(c, s) for s in SEEDS} for c in CONDITIONS}

    out_records = {}
    for w_str in tested_words:
        word_ops = parse_word(w_str)
        truth_trajectory = truth_trajectory_for(word_ops, test_ic)
        truth_end = np.asarray(truth_trajectory[:, -1], np.float64)
        seq = word_sequence(word_ops)

        rec = {"length": len(word_ops), "conditions": {}}
        for cond in CONDITIONS:
            seed_results = []
            for s in SEEDS:
                surr = surrogate_rollout(seq, dicts[cond][s], test_ic, grid)
                traj_err = traj_rel_l2_frames(surr, truth_trajectory)
                end_err = relative_l2_2d(np.asarray(surr[:, -1], np.float64), truth_end)
                seed_results.append({
                    "seed": s,
                    "trajectory_rel_l2": float(np.mean(traj_err)),
                    "endpoint_rel_l2": float(np.median(end_err)),
                })
            rec["conditions"][cond] = seed_results

        # E3: test-time operator splitting with the NARROW dictionary, length-3 words.
        if len(word_ops) == 3:
            split_rows = []
            for s in SEEDS:
                row = {"seed": s}
                for name, scheme in splitting_schemes(word_ops).items():
                    st = surrogate_rollout(scheme, dicts["narrow"][s], test_ic, grid)
                    row[name] = float(np.median(relative_l2_2d(np.asarray(st[:, -1], np.float64), truth_end)))
                row["best_scheme"] = min(
                    (k for k in row if k not in ("seed", "best_scheme")), key=lambda k: row[k]
                )
                row["best_endpoint_rel_l2"] = row[row["best_scheme"]]
                split_rows.append(row)
            rec["testtime_splitting_narrow"] = split_rows

        out_records[w_str] = rec
        print(f"  {w_str:8s} evaluated.", flush=True)

    elapsed = time.perf_counter() - started
    payload = {
        "schema": "family8-words-v2",
        "tested_words": tested_words,
        "eval_units": int(EVAL_UNITS),
        "words": out_records,
        "elapsed_seconds": elapsed,
    }
    (RESULTS / "WORDS.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote WORDS.json to {RESULTS / 'WORDS.json'} ({elapsed:.1f}s)", flush=True)


# ============================================================================
# SUBCOMMAND: algebra
# ============================================================================
def algebra() -> None:
    """Algebra recovery: exhaustive composition search over all words of length <= 3."""
    started = time.perf_counter()
    print("=== RUNNING OPERATOR-ALGEBRA RECOVERY SEARCH ===", flush=True)

    grid = make_grid_2d(FULL_GRID)
    candidates = []
    for length in (1, 2, 3):
        for prod in itertools.product(MEMBERS, repeat=length):
            candidates.append("o".join(prod))
    cand_ops = [parse_word(c) for c in candidates]
    cand_len = [len(o) for o in cand_ops]
    print(f"Search space size: {len(candidates)} candidate words (length 1, 2, 3)", flush=True)

    tested_words = tested_word_list()
    test_ic = initial_conditions_2d(HELDOUT_UNITS, NARROW, MASTER_SEED + 99, FULL_GRID)
    num_heldout = int(test_ic.shape[0])
    n_leg = STEPS_PER_LEG // K

    # Truth trajectories for every tested word, computed once (they do not depend on
    # the dictionary). All words share the same held-out ICs, which is also what lets
    # each candidate rollout be computed once and scored against every truth word.
    truth = {w: truth_trajectory_for(parse_word(w), test_ic) for w in tested_words}
    t_max = max(int(v.shape[1]) for v in truth.values())

    per_seed = {c: [] for c in CONDITIONS}
    for cond in CONDITIONS:
        for s in SEEDS:
            models = load_dictionary(cond, s)
            losses = {w: np.zeros((len(candidates), num_heldout)) for w in tested_words}
            for ci, ops in enumerate(cand_ops):
                traj = surrogate_rollout(word_sequence(ops, n_leg), models, test_ic, grid)
                # A candidate shorter than the observed horizon is held at its final
                # state: it asserts that nothing further happens. A candidate longer
                # than the horizon is truncated at the horizon. No tunable penalty.
                if traj.shape[1] < t_max:
                    pad = jnp.repeat(traj[:, -1:], t_max - traj.shape[1], axis=1)
                    traj = jnp.concatenate([traj, pad], axis=1)
                for w, tt in truth.items():
                    tw = int(tt.shape[1])
                    losses[w][ci] = traj_rel_l2_frames(traj[:, :tw], tt)

            word_acc, order_acc, mset_acc = {}, {}, {}
            tot_top1 = tot_mset = tot_n = 0
            for w in tested_words:
                true_ops = parse_word(w)
                n_top1 = n_mset = 0
                for u in range(num_heldout):
                    col = losses[w][:, u]
                    # Occam tie-break: exact ties (a longer candidate sharing the whole
                    # observed prefix scores identically) resolve to the shorter word.
                    best = min(range(len(candidates)), key=lambda i: (col[i], cand_len[i]))
                    if candidates[best] == w:
                        n_top1 += 1
                    if sorted(cand_ops[best]) == sorted(true_ops):
                        n_mset += 1
                word_acc[w] = n_top1 / num_heldout
                mset_acc[w] = n_mset / num_heldout
                order_acc[w] = (n_top1 / n_mset) if n_mset else None
                tot_top1 += n_top1
                tot_mset += n_mset
                tot_n += num_heldout

            rec = {
                "seed": s,
                "mean_top1_word_acc": float(np.mean(list(word_acc.values()))),
                "multiset_acc": float(tot_mset / tot_n),
                "order_acc_given_multiset": (float(tot_top1 / tot_mset) if tot_mset else None),
                "order_acc_denominator": int(tot_mset),
                "per_word_acc": word_acc,
                "per_word_multiset_acc": mset_acc,
                "per_word_order_acc_given_multiset": order_acc,
            }
            per_seed[cond].append(rec)
            print(f"  {cond:7s} seed {s} | top-1 {rec['mean_top1_word_acc']:.3f} | "
                  f"multiset {rec['multiset_acc']:.3f}", flush=True)

    summary = {}
    for cond in CONDITIONS:
        accs = [r["mean_top1_word_acc"] for r in per_seed[cond]]
        summary[cond] = {
            "median_top1_acc": float(np.median(accs)),
            "mean_top1_acc": float(np.mean(accs)),
            "per_seed_top1_acc": accs,
            "seed0_top1_acc": accs[0],
            "median_order_acc_given_multiset": float(np.median(
                [r["order_acc_given_multiset"] for r in per_seed[cond]
                 if r["order_acc_given_multiset"] is not None] or [float("nan")])),
            "median_multiset_acc": float(np.median([r["multiset_acc"] for r in per_seed[cond]])),
        }

    broad_acc = summary["broadS"]["median_top1_acc"]
    narrow_acc = summary["narrow"]["median_top1_acc"]
    acc_gain = broad_acc - narrow_acc
    alg1_pass = bool(broad_acc >= ALG1_MIN_BROAD_ACC and acc_gain >= ALG1_MIN_ACC_GAIN)

    payload = {
        "schema": "family8-algebra-v2",
        "tested_words": tested_words,
        "candidates": len(candidates),
        "heldout_per_word": num_heldout,
        "scoring": ("mean over stride-K frames 1..T of relative L2; candidates shorter than "
                    "the observed horizon are held at their final state; exact ties resolve "
                    "to the shorter word"),
        "ALG1_gate": {
            "aggregation": "median over the 5 seeds",
            "broadS_top1_acc": broad_acc,
            "narrow_top1_acc": narrow_acc,
            "absolute_gain": acc_gain,
            "threshold_acc": ALG1_MIN_BROAD_ACC,
            "threshold_gain": ALG1_MIN_ACC_GAIN,
            "pass": alg1_pass,
            "seed0_broadS_top1_acc": summary["broadS"]["seed0_top1_acc"],
            "seed0_narrow_top1_acc": summary["narrow"]["seed0_top1_acc"],
        },
        "summary": summary,
        "per_seed": per_seed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "ALGEBRA.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nALGEBRA.json written: ALG-1 pass={alg1_pass} "
          f"(broadS={broad_acc:.3f}, narrow={narrow_acc:.3f}, gain={acc_gain:+.3f})", flush=True)


# ============================================================================
# SUBCOMMAND: external
# ============================================================================
def external() -> None:
    """Secondary external-data arm: PDEBench 2-D diffusion-reaction (no gate).

    The arm is bounded on purpose. It records exactly what it did: the DaRUS dataset
    it queried, the files that dataset holds with their sizes, what it downloaded, and
    the outcome. Anything it could not do is recorded as NOT ATTEMPTED with the
    operational reason. Nothing here is ever synthesised.
    """
    started = time.perf_counter()
    print("=== EXTERNAL-DATA VALIDATION (PDEBench 2-D diffusion-reaction) ===", flush=True)

    import urllib.request
    import urllib.error

    DOI = "doi:10.18419/darus-2986"  # PDEBench, DaRUS
    API = ("https://darus.uni-stuttgart.de/api/datasets/:persistentId/"
           f"?persistentId={DOI}")
    MAX_BYTES = 3 * 1024 ** 3
    MAX_SECONDS = 900

    record = {
        "schema": "family8-external-v2",
        "dataset": "PDEBench 2D_diffusion-reaction (FitzHugh-Nagumo, Du=1e-3, Dv=5e-3, k=5e-3)",
        "source_doi": DOI,
        "api_query": API,
        "network": None,
        "files_seen": [],
        "download": None,
        "reproduction_check": {"status": "NOT ATTEMPTED", "reason": None},
        "algebra_on_external": {"status": "NOT ATTEMPTED", "reason": None},
        "status": "NOT ATTEMPTED",
        "reason": None,
    }

    try:
        with urllib.request.urlopen(API, timeout=60) as resp:
            meta = json.loads(resp.read().decode())
        record["network"] = "reachable"
    except Exception as exc:  # noqa: BLE001
        record["network"] = "unreachable"
        record["reason"] = f"DaRUS dataset API not reachable from the pod: {type(exc).__name__}: {exc}"
        record["reproduction_check"]["reason"] = record["reason"]
        record["algebra_on_external"]["reason"] = record["reason"]
        record["elapsed_seconds"] = time.perf_counter() - started
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "EXTERNAL.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Wrote EXTERNAL.json (NOT ATTEMPTED): {record['reason']}", flush=True)
        return

    files = meta.get("data", {}).get("latestVersion", {}).get("files", [])
    cands = []
    for f in files:
        df = f.get("dataFile", {})
        name = df.get("filename", "")
        if "diff-react" in name.lower() or "diffusion-reaction" in name.lower():
            cands.append({"filename": name, "size_bytes": df.get("filesize"), "id": df.get("id")})
    record["files_seen"] = cands
    print(f"  DaRUS reachable; {len(files)} files in dataset, "
          f"{len(cands)} match 2-D diffusion-reaction", flush=True)
    for c in cands:
        print(f"    {c['filename']}  {c['size_bytes']} bytes", flush=True)

    small = [c for c in cands if c["size_bytes"] and c["size_bytes"] <= MAX_BYTES]
    if not cands:
        record["reason"] = ("No file whose name identifies the 2-D diffusion-reaction set was "
                            f"listed under {DOI}.")
    elif not small:
        sizes = ", ".join(f"{c['filename']}={c['size_bytes'] / 1024 ** 3:.1f} GiB" for c in cands)
        record["reason"] = (f"Every 2-D diffusion-reaction file exceeds the {MAX_BYTES / 1024 ** 3:.0f} GiB "
                            f"download cap set for this run's GPU budget ({sizes}).")
    else:
        target = min(small, key=lambda c: c["size_bytes"])
        url = f"https://darus.uni-stuttgart.de/api/access/datafile/{target['id']}"
        dest = WORK / "external" / target["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                n = 0
                while True:
                    chunk = resp.read(1 << 22)
                    if not chunk:
                        break
                    fh.write(chunk)
                    n += len(chunk)
                    if time.perf_counter() - t0 > MAX_SECONDS:
                        raise TimeoutError(
                            f"download exceeded {MAX_SECONDS}s after {n} of {target['size_bytes']} bytes")
            record["download"] = {"url": url, "bytes": n,
                                  "seconds": time.perf_counter() - t0, "path": str(dest)}
            print(f"  downloaded {n} bytes in {time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            record["download"] = {"url": url, "error": f"{type(exc).__name__}: {exc}",
                                  "seconds": time.perf_counter() - t0}
            record["reason"] = f"Download of {target['filename']} failed: {type(exc).__name__}: {exc}"
            if dest.exists():
                dest.unlink()

        if record["download"] and "error" not in record["download"]:
            try:
                import h5py  # noqa: PLC0415
                with h5py.File(dest, "r") as h5:
                    keys = list(h5.keys())
                    probe = keys[0]
                    grp = h5[probe]
                    shapes = {k: list(np.asarray(grp[k]).shape) for k in list(grp.keys())[:6]} \
                        if hasattr(grp, "keys") else {}
                record["dataset_structure"] = {"top_level_keys": keys[:8],
                                               "n_top_level": len(keys),
                                               "first_group": probe, "shapes": shapes}
                record["reason"] = None
            except ModuleNotFoundError:
                record["reason"] = ("h5py is not in the pinned environment, so the downloaded "
                                    "HDF5 file could not be opened.")
            except Exception as exc:  # noqa: BLE001
                record["reason"] = f"Opening the downloaded file failed: {type(exc).__name__}: {exc}"

    # The two validation sub-parts are recorded honestly whatever happened above.
    why = ("Exponax 0.2.0 ships no two-species FitzHugh-Nagumo stepper, and PDEBench's "
           "2-D diffusion-reaction data is generated with no-flow Neumann boundaries on "
           "[-1,1]^2 while every stepper in this family is periodic and spectral. A "
           "matched reproduction therefore needs a new two-species solver with "
           "non-periodic boundaries, which is outside this run's declared scope and "
           "budget. Recorded as NOT ATTEMPTED rather than approximated.")
    record["reproduction_check"]["reason"] = record["reproduction_check"]["reason"] or why
    record["algebra_on_external"]["reason"] = record["algebra_on_external"]["reason"] or (
        "Depends on the matched dictionary from the reproduction check above.")
    if record["reason"] is None:
        record["reason"] = why
    record["status"] = "NOT ATTEMPTED"
    record["elapsed_seconds"] = time.perf_counter() - started

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "EXTERNAL.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote EXTERNAL.json ({record['status']}).", flush=True)


# ============================================================================
# SUBCOMMAND: gates
# ============================================================================
def gates() -> None:
    """Verify all 40 pregate files and check G1, G2 (blocking) plus imported 1-D diagnostics."""
    started = time.perf_counter()
    print("=== EVALUATING FAMILY 8 GATES (pre-off-support) ===", flush=True)

    pregate_files = {}
    for cond in CONDITIONS:
        pregate_files[cond] = {}
        for m in MEMBERS:
            pregate_files[cond][m] = {}
            for s in SEEDS:
                fpath = RESULTS / cond / f"{m}_seed{s}_pregate.json"
                if not fpath.exists():
                    raise SystemExit(f"Missing pregate file: {fpath}. All 40 fits required for gates.")
                pregate_files[cond][m][s] = json.loads(fpath.read_text())

    narrow_e_ons = [pregate_files["narrow"][m][s]["e_on"] for m in MEMBERS for s in SEEDS]
    broad_e_ons = [pregate_files["broadS"][m][s]["e_on"] for m in MEMBERS for s in SEEDS]
    min_all_e_on = float(min(min(narrow_e_ons), min(broad_e_ons)))
    worst_broad_e_on = float(max(broad_e_ons))

    # G2 is per member: every member's surrogate must resolve its own operator, i.e.
    # its on-support error must be well below how much that operator moves the state
    # over the estimand stride.
    g2_details = {}
    for m in MEMBERS:
        ec = eval_cohorts_2d(m, FULL_GRID)
        pers_on = float(np.median(relative_l2_2d(
            np.asarray(ec["on_states"], np.float64), np.asarray(ec["on_target"], np.float64))))
        med_nrw = float(np.median([pregate_files["narrow"][m][s]["e_on"] for s in SEEDS]))
        med_brd = float(np.median([pregate_files["broadS"][m][s]["e_on"] for s in SEEDS]))
        g2_details[m] = {
            "persistence_on": pers_on,
            "median_narrow_e_on": med_nrw,
            "median_broadS_e_on": med_brd,
            "e_on_limit": G2_MAX_E_ON_FRACTION * pers_on,
            "pass": bool(med_nrw <= G2_MAX_E_ON_FRACTION * pers_on),
        }
        print(f"  G2 {m}: e_on(narrow)={med_nrw:.4e} vs limit {G2_MAX_E_ON_FRACTION * pers_on:.4e} "
              f"(persistence {pers_on:.4e}) -> {g2_details[m]['pass']}", flush=True)
    g2_pass = all(v["pass"] for v in g2_details.values())

    ps_file = RESULTS / "PRESCREEN.json"
    if not ps_file.exists():
        raise SystemExit("Missing PRESCREEN.json. Run prescreen first.")
    ps_data = json.loads(ps_file.read_text())
    g1_pass = len(ps_data.get("passing_pairs", [])) > 0

    blocking = []
    if not g1_pass:
        blocking.append("G1_shift_exists")
    if not g2_pass:
        blocking.append("G2_metric_resolves")

    payload = {
        "schema": "family8-gates-v2",
        "written_before_any_off_support_number_was_read": True,
        "prescreen_grid": ps_data.get("num_points"),
        "G1_shift_exists": {
            "pass": g1_pass,
            "passing_pairs": ps_data.get("passing_pairs", []),
            "passing_pairs_count": len(ps_data.get("passing_pairs", [])),
            "blocking": True,
        },
        "G2_metric_resolves": {"pass": g2_pass, "per_member": g2_details, "blocking": True},
        "imported_1d_reference_diagnostics": {
            "note": ("G4a and P6 carry absolute e_on constants taken from the 1-D families "
                     "(family 1 / 1b, 1-D FNO at 256 points). They are reported against the "
                     "2-D fits for the record and are NOT blocking here, because no 2-D "
                     "calibration of those two constants exists. See REVIEW_NOTES.md item 6."),
            "G4a_censored_below": {"min_all_e_on": min_all_e_on, "floor_1d": G4_MIN_E_ON,
                                   "meets_1d_reference": bool(min_all_e_on >= G4_MIN_E_ON)},
            "P6_convergence": {"worst_broad_e_on": worst_broad_e_on, "limit_1d": P6_MAX_E_ON,
                               "meets_1d_reference": bool(worst_broad_e_on <= P6_MAX_E_ON)},
        },
        "e_on_median_by_member": {
            c: {m: float(np.median([pregate_files[c][m][s]["e_on"] for s in SEEDS])) for m in MEMBERS}
            for c in CONDITIONS
        },
        "blocking_failures": blocking,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"GATES.json written: G1={g1_pass}, G2={g2_pass}, blocking={blocking}", flush=True)


# ============================================================================
# SUBCOMMAND: evaluate
# ============================================================================
def evaluate() -> None:
    """Aggregate all results and evaluate endpoints E1, E2, E3."""
    gates_f = RESULTS / "GATES.json"
    if not gates_f.exists():
        raise SystemExit("GATES.json missing: refusing to evaluate")
    g_data = json.loads(gates_f.read_text())
    if g_data["blocking_failures"]:
        raise SystemExit(f"Blocking failures present: {g_data['blocking_failures']}")

    started = time.perf_counter()
    print("=== FINAL EVALUATION: FAMILY 8 ===", flush=True)

    words_f = RESULTS / "WORDS.json"
    alg_f = RESULTS / "ALGEBRA.json"
    if not words_f.exists():
        raise SystemExit("WORDS.json missing: run words first")
    if not alg_f.exists():
        raise SystemExit("ALGEBRA.json missing: run algebra first")

    words_data = json.loads(words_f.read_text())
    alg_data = json.loads(alg_f.read_text())

    pregate = {c: {m: {s: json.loads((RESULTS / c / f"{m}_seed{s}_pregate.json").read_text())
                       for s in SEEDS} for m in MEMBERS} for c in CONDITIONS}
    e_on = {c: {m: {s: pregate[c][m][s]["e_on"] for s in SEEDS} for m in MEMBERS} for c in CONDITIONS}
    fit_seconds = [pregate[c][m][s]["train_seconds"] for c in CONDITIONS for m in MEMBERS for s in SEEDS]

    # E1: degradation R = composed-rollout endpoint error / on-support error of the
    # participating surrogates (median over the word's distinct members, per seed).
    per_word = {}
    ratios_R, ratios_raw = [], []
    for w_str, w_dict in words_data["words"].items():
        members_w = sorted(set(parse_word(w_str)))
        row = {"length": w_dict["length"], "members": members_w}
        for cond in CONDITIONS:
            rows = w_dict["conditions"][cond]
            ends = [r["endpoint_rel_l2"] for r in rows]
            trajs = [r["trajectory_rel_l2"] for r in rows]
            R = [r["endpoint_rel_l2"] / float(np.median([e_on[cond][m][r["seed"]] for m in members_w]))
                 for r in rows]
            row[cond] = {
                "median_endpoint_rel_l2": float(np.median(ends)),
                "median_trajectory_rel_l2": float(np.median(trajs)),
                "median_R": float(np.median(R)),
                "per_seed_R": R,
                "per_seed_endpoint_rel_l2": ends,
            }
        row["P3_ratio_R"] = row["broadS"]["median_R"] / max(row["narrow"]["median_R"], 1e-12)
        row["P3_ratio_endpoint_unnormalised"] = (
            row["broadS"]["median_endpoint_rel_l2"] / max(row["narrow"]["median_endpoint_rel_l2"], 1e-12))
        row["pass_p3"] = bool(row["P3_ratio_R"] <= P3_MAX_BROAD_FRACTION)
        ratios_R.append(row["P3_ratio_R"])
        ratios_raw.append(row["P3_ratio_endpoint_unnormalised"])
        per_word[w_str] = row

    agg_repair_ratio = float(np.median(ratios_R))
    p3_pass = bool(agg_repair_ratio <= P3_MAX_BROAD_FRACTION)

    # E3: test-time operator splitting (narrow dictionary) on the length-3 words.
    splitting = {}
    for w_str, w_dict in words_data["words"].items():
        rows = w_dict.get("testtime_splitting_narrow")
        if not rows:
            continue
        best = [r["best_endpoint_rel_l2"] for r in rows]
        scheme_names = [k for k in rows[0] if k not in ("seed", "best_scheme", "best_endpoint_rel_l2")]
        splitting[w_str] = {
            "per_scheme_median_endpoint_rel_l2": {
                k: float(np.median([r[k] for r in rows])) for k in scheme_names},
            "best_scheme_per_seed": [r["best_scheme"] for r in rows],
            "median_best_endpoint_rel_l2": float(np.median(best)),
            "median_narrow_sequential_endpoint_rel_l2": per_word[w_str]["narrow"]["median_endpoint_rel_l2"],
            "median_broadS_sequential_endpoint_rel_l2": per_word[w_str]["broadS"]["median_endpoint_rel_l2"],
            "broadS_over_best_splitting": (per_word[w_str]["broadS"]["median_endpoint_rel_l2"]
                                           / max(float(np.median(best)), 1e-12)),
        }

    ext_f = RESULTS / "EXTERNAL.json"
    external_status = json.loads(ext_f.read_text()) if ext_f.exists() else {"status": "NOT RUN"}

    alg1_pass = alg_data["ALG1_gate"]["pass"]
    verdict = "PASS" if (p3_pass and alg1_pass) else "FAIL"

    final_record = {
        "schema": "family8-result-v2",
        "verdict": verdict,
        "tested_words": words_data["tested_words"],
        "grid": [FULL_GRID, FULL_GRID],
        "seeds": list(SEEDS),
        "n_train_units": NUM_TRAIN_UNITS,
        "fno_parameters": count_parameters(FNO2d(key=jr.PRNGKey(0))),
        "per_fit_seconds": {
            "median": float(np.median(fit_seconds)),
            "min": float(np.min(fit_seconds)),
            "max": float(np.max(fit_seconds)),
            "n_fits": len(fit_seconds),
        },
        "E1_P3_broadening_repairs": {
            "aggregated_repair_ratio_R": agg_repair_ratio,
            "aggregated_repair_ratio_endpoint_unnormalised": float(np.median(ratios_raw)),
            "threshold": P3_MAX_BROAD_FRACTION,
            "pass": p3_pass,
            "per_word": per_word,
        },
        "E2_ALG1_algebra_recovery": alg_data["ALG1_gate"],
        "E2_summary": alg_data["summary"],
        "E3_testtime_splitting_narrow": splitting,
        "e_on_median_by_member": g_data["e_on_median_by_member"],
        "external": external_status,
        "gates": {"blocking_failures": g_data["blocking_failures"],
                  "G2_per_member": g_data["G2_metric_resolves"]["per_member"],
                  "imported_1d_reference_diagnostics": g_data["imported_1d_reference_diagnostics"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "RESULT.json").write_text(json.dumps(final_record, indent=2) + "\n")
    print(f"\nRESULT.json written: Verdict={verdict} | P3 ratio={agg_repair_ratio:.4f} "
          f"(<=0.25: {p3_pass}) | ALG-1 pass={alg1_pass}", flush=True)


# ============================================================================
# Main CLI Entrypoint
# ============================================================================
def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        raise SystemExit(
            "usage: run_family8.py prescreen | smoke [steps] | train <member> <seed> <cond> | "
            "words | algebra | external | gates | evaluate"
        )
    cmd = argv[0]
    if cmd == "prescreen":
        prescreen(int(argv[1]) if len(argv) > 1 else FULL_GRID)
    elif cmd == "smoke":
        steps = int(argv[1]) if len(argv) > 1 else SMOKE_STEPS
        smoke(steps)
    elif cmd == "train":
        if len(argv) < 4:
            raise SystemExit("usage: run_family8.py train <member> <seed> <cond>")
        train(argv[1], int(argv[2]), argv[3])
    elif cmd == "words":
        words()
    elif cmd == "algebra":
        algebra()
    elif cmd == "external":
        external()
    elif cmd == "gates":
        gates()
    elif cmd == "evaluate":
        evaluate()
    else:
        raise SystemExit(f"Unknown command {cmd!r}")


if __name__ == "__main__":
    main()
