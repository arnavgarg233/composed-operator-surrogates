"""POST-HOC mechanism diagnostic. Gates nothing; explains why statistic S behaves as it does.

Declared here as post-hoc: it was written AFTER results/SUPPORT.json was read. It is not
a third support statistic and it is not substituted for S or S2 anywhere. It answers one
question: how much of a switch state's deviation from the training mean lies in the
directions the PCA-99 basis DISCARDS (the residual / Q direction, which the package's own
state_support table also reports as `residual_l2` beside `mahalanobis`)?

Writes results/DIAGNOSTIC.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rf1c", HERE / "run_family1c.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import timing_probe as tp  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    switch = {p: m.eval_cohorts(p)["switch_states"] for p in ("P1", "P2", "CH")}
    out = {"schema": "family1c-posthoc-diagnostic-v1",
           "status": "POST-HOC; gates nothing; written after SUPPORT.json was read",
           "cells": {}}
    groups = [("narrow", ["P1", "P2", "CH"]), ("broad", ["P1"]), ("broad", ["P2"]),
              ("broad", ["CH"]), ("broadS", ["CH"]), ("broadS", ["P1"])]
    for condition, pairs in groups:
        ts = m.train_states_for(pairs[0], condition)
        x = np.asarray(ts, np.float64).reshape(-1, tp.NUM_POINTS)
        mu = x.mean(axis=0)
        xc = x - mu
        n = x.shape[0]
        cov = (xc.T @ xc) / (n - 1)
        lam, vec = np.linalg.eigh(cov)
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]
        k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.99) + 1)
        u = vec[:, :k]
        res_tr = np.linalg.norm(xc - (xc @ u) @ u.T, axis=1)
        q99 = float(np.percentile(res_tr, 99))
        for p in pairs:
            s = np.asarray(switch[p], np.float64).reshape(-1, tp.NUM_POINTS)
            sc = s - mu
            res_sw = np.linalg.norm(sc - (sc @ u) @ u.T, axis=1)
            tot = np.linalg.norm(sc, axis=1)
            out["cells"][f"{p}_{condition}"] = {
                "pca_k_99": k,
                "train_residual_p99": q99,
                "switch_residual_mean": float(res_sw.mean()),
                "switch_residual_over_train_p99_median": float(
                    np.median(res_sw) / q99),
                "switch_fraction_of_deviation_outside_basis_median": float(
                    np.median(res_sw / tot)),
                "coverage_inside_train_residual_p99": float((res_sw <= q99).mean()),
            }
            r = out["cells"][f"{p}_{condition}"]
            print(f"{p}_{condition:8} k={k:3d} Q99 {q99:8.4f} "
                  f"switch Q mean {res_sw.mean():9.4f} "
                  f"Q/Q99 median {r['switch_residual_over_train_p99_median']:8.2f} "
                  f"inside {r['coverage_inside_train_residual_p99']:.4f}", flush=True)
        del ts, x, xc
    out["elapsed_seconds"] = time.perf_counter() - started
    (m.RELEASE_ROOT / "results" / "family1c_state_support" / "DIAGNOSTIC.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(f"wrote DIAGNOSTIC.json ({out['elapsed_seconds']:.1f} s)")


if __name__ == "__main__":
    main()
