"""Family 4b: U-Net BROAD-S on P1, with a smoke-safe full-run driver.

The architecture is imported from family 4 and the BROAD-S block schedule is
imported from family 1e.  The narrow arm is deliberately not refit: its
per-seed R values are read from family 4's completed P1 result.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path("pde_volume/family4b")
RESULTS = WORK / "results"
CKPT = WORK / "ckpt"
LOGS = WORK / "logs"

F4 = Path("pde_restart/family4_second_architecture/run_family4.py")
F1E = Path("pde_restart/family1e_state_broadening_new_pairs/run_family1e.py")
F4_RESULT = Path("pde_volume/family4/results/P1_diffusion_fisher_kpp/RESULT.json")


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Read-only imports.  f4 supplies UNet1D, optimizer, pair construction and
# evaluation; f1e supplies the predeclared BROAD-S block schedule.
f4 = _import(F4, "family4_readonly")
f1e = _import(F1E, "family1e_readonly")
tp, rb = f4.tp, f4.rb

PAIR = "P1_diffusion_fisher_kpp"
SEEDS = tuple(range(10))
STEPS = 8000
BATCH = 64
SMOKE_STEPS = 200
N_TRAIN = 640
N_EVAL = 256
NARROW = (0.4865, 0.5171)
BLOCKS = f1e.BROADS_BLOCKS
ARCH_PARITY_LIMIT = 6.8648929485e-4


def _second_operator():
    return tp.build_steppers(tp.DT)[1]


def _diffusion():
    return tp.build_steppers(tp.DT)[0]


def _grid():
    return jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)


def broadS_states():
    """Family 1e sampler, specialized to P1's declared second operator."""
    base = tp.initial_conditions(N_TRAIN, NARROW, tp.MASTER_SEED + 1)
    traj = tp.rollout(_second_operator(), base, tp.STEPS_PER_TRAJECTORY)
    ic = jnp.concatenate([traj[lo:hi, m] for lo, hi, m in BLOCKS], axis=0)
    del traj
    return tp.rollout(_diffusion(), ic, tp.STEPS_PER_TRAJECTORY)


def eval_cohorts():
    d = _diffusion()
    on_ic = tp.initial_conditions(N_EVAL, NARROW, tp.MASTER_SEED + 7)
    on = tp.rollout(d, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(d, on, f4.K)[:, -1]
    sw_ic = tp.initial_conditions(N_EVAL, NARROW, tp.MASTER_SEED + 8)
    sw = tp.rollout(_second_operator(), sw_ic, tp.STEPS_PER_TRAJECTORY)[:, -1]
    sw_target = tp.rollout(d, sw, f4.K)[:, -1]
    return on, on_target, sw, sw_target


def _relative(pred, target):
    return np.sqrt(((pred - target) ** 2).sum(axis=(1, 2))) / np.sqrt(
        (target ** 2).sum(axis=(1, 2)))


def _evaluate(model, states, target, grid):
    out = []
    for start in range(0, len(states), BATCH):
        out.append(np.asarray(f4.model_apply(model, states[start:start + BATCH], grid)))
    return _relative(np.concatenate(out), np.asarray(target))


def _save(path, model):
    leaves, _ = jax.tree_util.tree_flatten(model)
    np.savez(path, **{f"l{i}": np.asarray(x) for i, x in enumerate(leaves)
                       if hasattr(x, "shape")})


def _load(seed: int):
    model = f4.UNet1D(jr.PRNGKey(seed))
    leaves, treedef = jax.tree_util.tree_flatten(model)
    with np.load(CKPT / PAIR / f"seed{seed}_broadS.npz") as data:
        arrays = [jnp.asarray(data[f"l{i}"]) for i in range(len(leaves))]
    return jax.tree_util.tree_unflatten(treedef, arrays)


def train(seed: int, steps: int = STEPS):
    if seed not in SEEDS:
        raise SystemExit(f"seed must be one of {SEEDS}")
    started = time.perf_counter()
    states = broadS_states()
    x, y = f4.pairs(states)
    del states
    grid = _grid()
    model = f4.UNet1D(jr.PRNGKey(seed))
    opt = f4.adam_init(model)
    key = jr.PRNGKey(seed + 9999)
    for step in range(steps):
        key, bk = jr.split(key)
        ix = jr.randint(bk, (BATCH,), 0, x.shape[0])
        lr = f4.LR_LOW + 0.5 * (f4.LR_HIGH - f4.LR_LOW) * (
            1 + np.cos(np.pi * step / STEPS))
        model, opt, _ = f4.train_step(model, opt, x[ix], y[ix], grid, lr)
    del x, y, opt
    on, on_target, _, _ = eval_cohorts()
    e_on = float(np.median(_evaluate(model, on, on_target, grid)))
    ckpt = CKPT / PAIR / f"seed{seed}_broadS.npz"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    _save(ckpt, model)
    out = {
        "schema": "family4b-broadS-pregate-v1", "pair": PAIR,
        "condition": "broadS", "seed": seed, "steps": steps, "batch": BATCH,
        "sampler": "family1e.BROADS_BLOCKS", "blocks": [list(x) for x in BLOCKS],
        "n_train": N_TRAIN, "e_on": e_on, "params": f4.count(model),
        "checkpoint": str(ckpt), "elapsed_seconds": time.perf_counter() - started,
        "note": "e_off is deliberately absent; gates must be written first",
    }
    RESULTS.joinpath(PAIR).mkdir(parents=True, exist_ok=True)
    (RESULTS / PAIR / f"seed{seed}_broadS_pregate.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(f"{PAIR} broadS seed {seed}: e_on={e_on:.6e} params={f4.count(model)} "
          f"steps={steps}", flush=True)
    return model, out


def gates():
    """Write pre-off-support gates using family 4's narrow arm."""
    old = json.loads(F4_RESULT.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(
        (RESULTS / PAIR).glob("seed*_broadS_pregate.json"))]
    if {r["seed"] for r in rows} != set(SEEDS):
        raise SystemExit("gates require all ten BROAD-S pregate files")
    narrow_rows = [r for r in old["per_seed"] if r["condition"] == "narrow"]
    narrow_by_seed = {r["seed"]: r for r in narrow_rows}
    broad_by_seed = {r["seed"]: r for r in rows}
    narrow_R = [narrow_by_seed[s]["R"] for s in SEEDS]
    broad_e = [broad_by_seed[s]["e_on"] for s in SEEDS]
    median_narrow_e = float(np.median([narrow_by_seed[s]["e_on"] for s in SEEDS]))
    on, on_target, sw, sw_target = eval_cohorts()
    persistence_on = float(np.median(_relative(np.asarray(on), np.asarray(on_target))))
    persistence_off = float(np.median(_relative(np.asarray(sw), np.asarray(sw_target))))
    f4_gates = F4_RESULT.with_name("GATES.json")
    geometry = (old.get("geometry")
                or (json.loads(f4_gates.read_text())["geometry"]
                    if f4_gates.exists() else f4.geometry(PAIR)))
    payload = {
        "schema": "family4b-broadS-gates-v1",
        "written_before_any_off_support_number_was_read": True,
        "pair": PAIR, "sampler": "family1e.BROADS_BLOCKS",
        "geometry": geometry,
        "G1": geometry["overlap"] == 0.0,
        "G2": persistence_off / median_narrow_e >= 100.0
                and median_narrow_e <= 0.1 * persistence_on,
        "G4_below": min([narrow_by_seed[s]["e_on"] for s in SEEDS] + broad_e)
                     >= 1.19e-5,
        "P6": max(broad_e) <= 1.573e-3,
        "ARCH-PARITY": median_narrow_e <= ARCH_PARITY_LIMIT,
        "persistence_on": persistence_on, "persistence_off": persistence_off,
        "median_e_on_narrow_reused": median_narrow_e,
        "narrow_R_reused_from_family4": narrow_R,
        "per_seed_e_on_broadS": {str(s): broad_by_seed[s]["e_on"] for s in SEEDS},
        "blocking_failures": [],
    }
    payload["blocking_failures"] = [k for k in
        ("G1", "G2", "G4_below", "P6", "ARCH-PARITY") if not payload[k]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def evaluate():
    gate = json.loads((RESULTS / "GATES.json").read_text())
    if gate["blocking_failures"]:
        raise SystemExit("blocking pre-off gate failure; refusing to read e_off")
    on, on_target, sw, sw_target = eval_cohorts()
    grid = _grid()
    old = json.loads(F4_RESULT.read_text())
    narrow_by_seed = {r["seed"]: r for r in old["per_seed"]
                      if r["condition"] == "narrow"}
    out_rows = []
    for seed in SEEDS:
        model = _load(seed)
        e_off = float(np.median(_evaluate(model, sw, sw_target, grid)))
        pred = np.asarray(f4.model_apply(model, sw, grid))
        corrected = float(np.median(_relative(
            pred - pred.mean(axis=-1, keepdims=True)
            + np.asarray(sw).mean(axis=-1, keepdims=True),
            np.asarray(sw_target))))
        e_on = json.loads((RESULTS / PAIR /
                           f"seed{seed}_broadS_pregate.json").read_text())["e_on"]
        out_rows.append({"seed": seed, "condition": "broadS", "e_on": e_on,
                         "e_off": e_off, "R": e_off / e_on,
                         "e_off_corrected": corrected,
                         "narrow_R_reused": narrow_by_seed[seed]["R"]})
    broad_R = [r["R"] for r in out_rows]
    narrow_R = [r["narrow_R_reused"] for r in out_rows]
    result = {
        "schema": "family4b-result-v1", "pair": PAIR, "per_seed": out_rows,
        "narrow_reused": {"median_R": float(np.median(narrow_R)),
                           "R_range": [min(narrow_R), max(narrow_R)],
                           "source": str(F4_RESULT)},
        "broadS": {"median_R": float(np.median(broad_R)),
                   "R_range": [min(broad_R), max(broad_R)]},
        "P3_broadening_repairs": {
            "held": bool(np.median(broad_R) <= 0.25 * np.median(narrow_R)),
            "ratio_broadS_to_narrow": float(np.median(broad_R) / np.median(narrow_R)),
        },
        "P4_constant_mode": {
            "median_corrected_e_off": float(np.median(
                [r["e_off_corrected"] for r in out_rows])),
            "median_e_off": float(np.median([r["e_off"] for r in out_rows])),
        },
        "G4_above": {"max_e_off": max(r["e_off"] for r in out_rows),
                      "persistence_off": gate["persistence_off"],
                      "pass": max(r["e_off"] for r in out_rows)
                      < gate["persistence_off"]},
    }
    (RESULTS / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def smoke(steps: int = SMOKE_STEPS):
    started = time.perf_counter()
    model, pre = train(0, steps=steps)
    states = broadS_states()
    grid = _grid()
    x, y = f4.pairs(states)
    del x, y, states
    smoke_out = {"schema": "family4b-smoke-v1", "pair": PAIR, "seed": 0,
                 "steps": steps, "params": f4.count(model), "e_on": pre["e_on"],
                 "sampler_blocks": [list(x) for x in BLOCKS],
                 "train_states_shape": [N_TRAIN, tp.STEPS_PER_TRAJECTORY + 1, 1,
                                         tp.NUM_POINTS],
                 "elapsed_seconds": time.perf_counter() - started,
                 "status": "PASS: broadS corpus, fit, evaluation and checkpoint"}
    (WORK / "SMOKE.json").write_text(json.dumps(smoke_out, indent=2) + "\n")
    print(json.dumps(smoke_out, indent=2), flush=True)


def main(argv):
    if not argv:
        raise SystemExit("usage: smoke [steps] | train <seed> | gates | evaluate")
    if argv[0] == "smoke":
        smoke(int(argv[1]) if len(argv) > 1 else SMOKE_STEPS)
    elif argv[0] == "train":
        train(int(argv[1]))
    elif argv[0] == "gates":
        gates()
    elif argv[0] == "evaluate":
        evaluate()
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
