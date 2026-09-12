"""Thin orchestration wrapper for the validated family-1 harness."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
HARNESS_PATH = RELEASE_ROOT / "dependency" / "family1_second_pairs" / "run_family1.py"
RESULTS = RELEASE_ROOT / "results" / "family1_rerun"
CKPT = RELEASE_ROOT / "work" / "family1_rerun" / "ckpt"

spec = importlib.util.spec_from_file_location("family1_harness", HARNESS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
rbh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbh)
rbh.RESULTS = RESULTS
rbh.CKPT = CKPT

PAIRS = ("S1_fkpp_r0.5", "S1_grayscott_1sp")
INTERVALS = {
    PAIRS[0]: {"narrow": [0.4865, 0.5171], "broad": [0.48, 0.70]},
    PAIRS[1]: {"narrow": [0.4865, 0.5171], "broad": [0.30, 0.52]},
}


def _reaction(cls, dt, **kw):
    return cls(rbh.tp.NUM_SPATIAL_DIMS, rbh.tp.DOMAIN_EXTENT, rbh.tp.NUM_POINTS, dt,
               order=rbh.tp.ETDRK_ORDER,
               dealiasing_fraction=rbh.tp.DEALIASING_FRACTION,
               num_circle_points=rbh.tp.NUM_CIRCLE_POINTS,
               circle_radius=rbh.tp.CIRCLE_RADIUS, **kw)


def second_operator(pair: str):
    if pair == PAIRS[0]:
        return _reaction(rbh.ex.stepper.reaction.FisherKPP, rbh.tp.DT,
                         diffusivity=0.0, reactivity=0.5)
    if pair == PAIRS[1]:
        F = 0.04
        return rbh.ex.stepper.generic.GeneralPolynomialStepper(
            rbh.tp.NUM_SPATIAL_DIMS, rbh.tp.DOMAIN_EXTENT, rbh.tp.NUM_POINTS, rbh.tp.DT,
            linear_coefficients=(-(1.0 + F), 0.0, 0.0),
            polynomial_coefficients=(F, 0.0, 2.0, -1.0),
            order=rbh.tp.ETDRK_ORDER,
            dealiasing_fraction=rbh.tp.DEALIASING_FRACTION,
            num_circle_points=rbh.tp.NUM_CIRCLE_POINTS,
            circle_radius=rbh.tp.CIRCLE_RADIUS)
    raise SystemExit(f"unknown pair {pair!r}")


rbh.second_operator = second_operator
rbh.PAIRS = PAIRS
rbh.NEW_PAIRS = PAIRS


def prepare():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    (RESULTS / "INTERVALS.json").write_text(json.dumps(INTERVALS, indent=2) + "\n")
    (RESULTS / "GEOMETRY.json").write_text(json.dumps({
        "schema": "family1-rerun-predeclared-geometry-v1",
        "narrow_training_support": [0.4865763783454895, 0.5169978737831116],
        "pairs": {
            PAIRS[0]: {"switch_support": [0.5941062569618225, 0.6244125366210938],
                       "narrow_overlap": 0.0, "narrow_gap": 0.07710838317871094},
            PAIRS[1]: {"switch_support": [0.390663206577301, 0.42818015813827515],
                       "narrow_overlap": 0.0, "narrow_gap": -0.12681771814819336},
        },
        "source": "family1b_pair_screen/RESULT.json and PREDECLARED.md",
    }, indent=2) + "\n")


def call(cmd):
    rbh.main(cmd)


def driver():
    prepare()
    launch_path = HERE / "LAUNCH.json"
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        launch.update({"status": "running", "driver": str(Path(__file__).resolve()),
                       "log": str(HERE / "logs" / "driver.log")})
        launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    logs = HERE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "JAX_PLATFORMS": "cpu",
                "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"})
    script = str(HERE / "run_family1_rerun.py")

    # Each worker is a separate single-threaded process, while this remains the
    # single detached driver process requested by the task.
    for pair in PAIRS:
        for condition in ("narrow", "broad"):
            for seed in range(10):
                log = logs / f"train_{pair}_{condition}_seed{seed}.log"
                with log.open("w") as fh:
                    p = subprocess.run(
                        [sys.executable, "-u", script, "train", pair, condition, str(seed)],
                        stdout=fh, stderr=subprocess.STDOUT, env=env)
                if p.returncode:
                    raise SystemExit(f"training failed: {pair} {condition} seed {seed}")
        call(["gates", pair])
        gate = json.loads((RESULTS / pair / "GATES.json").read_text())
        if gate["blocking_failures"]:
            (RESULTS / pair / "RESULT.json").write_text(json.dumps({
                "pair": pair, "verdict": "FAIL",
                "blocking_failures": gate["blocking_failures"]}, indent=2) + "\n")
            continue
        for condition in ("narrow", "broad"):
            for seed in range(10):
                log = logs / f"eval_{pair}_{condition}_seed{seed}.log"
                with log.open("w") as fh:
                    p = subprocess.run(
                        [sys.executable, "-u", script, "eval", pair, condition, str(seed)],
                        stdout=fh, stderr=subprocess.STDOUT, env=env)
                if p.returncode:
                    raise SystemExit(f"evaluation failed: {pair} {condition} seed {seed}")
        call(["aggregate", pair])

    pair_results = {}
    for pair in PAIRS:
        path = RESULTS / pair / "RESULT.json"
        if path.exists():
            pair_results[pair] = json.loads(path.read_text())
    passed = all(r.get("P3_broadening_repairs", {}).get("held", False)
                 for r in pair_results.values()) and len(pair_results) == len(PAIRS)
    family = {"schema": "family1-rerun-result-v1", "pairs": pair_results,
              "verdict": "PASS" if passed else "FAIL",
              "rule": "PASS iff P3 holds on both pairs"}
    (RESULTS / "RESULT.json").write_text(json.dumps(family, indent=2) + "\n")
    md = ["# Family 1 rerun result", "", f"**Verdict: {family['verdict']}**", ""]
    for pair, result in pair_results.items():
        md.append(f"- `{pair}`: {result.get('verdict', {}).get('pair_verdict', 'FAIL')}; "
                  f"P3={'PASS' if result.get('P3_broadening_repairs', {}).get('held') else 'FAIL'}")
    (RESULTS / "RESULT.md").write_text("\n".join(md) + "\n")


def main(argv):
    if argv and argv[0] == "driver":
        driver()
    elif argv and argv[0] in {"train", "gates", "eval", "aggregate"}:
        call(argv)
    else:
        raise SystemExit("usage: run_family1_rerun.py driver|train|gates|eval|aggregate ...")


if __name__ == "__main__":
    main(sys.argv[1:])
