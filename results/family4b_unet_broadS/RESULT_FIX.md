# Result of Bug Fix & TASK.md Verification for Family 4b

## 1. Applied Diff

Added `sys.modules[name] = mod` immediately before `spec.loader.exec_module(mod)` in `run_family4b.py`:

```diff
--- run_family4b.py
+++ run_family4b.py
@@ -33,6 +33,7 @@
     spec = importlib.util.spec_from_file_location(name, path)
     mod = importlib.util.module_from_spec(spec)
     assert spec.loader is not None
+    sys.modules[name] = mod
     spec.loader.exec_module(mod)
     return mod
```

## 2. Import-Check Verification

Command executed:
```sh
source pde_volume/ENV.sh; JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 pde_restart/env/bin/python -c "import run_family4b as m; print(m.f4.count, m.BLOCKS, m.NARROW)"
```

Printed output:
```
<function count at 0x111ce5300> ((0, 160, 0), (160, 320, 25), (320, 480, 50), (480, 640, 100)) (0.4865, 0.5171)
```

## 3. Checklist: Cross-Check of `run_family4b.py` against `TASK.md`

| Item | Status | Line Numbers & Details |
| :--- | :---: | :--- |
| **Narrow $R$ read from family 4 RESULT.json (not refit)** | **MATCH** | Lines 29 (`F4_RESULT = Path("pde_volume/family4/results/P1_diffusion_fisher_kpp/RESULT.json")`), 157 (`old = json.loads(F4_RESULT.read_text())`), 162–165 (`narrow_rows = ...; narrow_R = [narrow_by_seed[s]["R"] for s in SEEDS]`), 186 (`"narrow_R_reused_from_family4": narrow_R`), 204–206, 221, 226–228 (`narrow_reused` block in `evaluate()`). Narrow condition is never refit. |
| **BROAD-S corpus identical to family 1e for P1** (same sampler, coverage target, $n=640$, MASTER_SEED derivation) | **MATCH** | Lines 43 (`f1e = _import(F1E, ...)`), 51 (`N_TRAIN = 640`), 53 (`NARROW = (0.4865, 0.5171)`), 54 (`BLOCKS = f1e.BROADS_BLOCKS`), 58–64 (`_second_operator()`, `_diffusion()`), 70–76 (`broadS_states()` rollouts from `tp.MASTER_SEED + 1` with `BLOCKS`), 79–87 (`eval_cohorts()` using `tp.MASTER_SEED + 7` and `+ 8`). Byte-identical construction to family 1e for P1. |
| **10 seeds** | **MATCH** | Lines 47 (`SEEDS = tuple(range(10))`), 117–118 (`seed not in SEEDS`), 160 (`{r["seed"] for r in rows} != set(SEEDS)`), 165–166, 187, 208 (`for seed in SEEDS:` in `evaluate()`), and `launch_driver.sh` line 11 (`for seed in $(seq 0 9)`). |
| **Gates G1, G2 $\ge$ 100, G4, P6, ARCH-PARITY within 3$\times$ FNO $e_{\text{on}}$, P3 $\le$ 0.25** | **MATCH** | Lines 55 (`ARCH_PARITY_LIMIT = 6.8648929485e-4`, which is $3 \times 2.2882976495\text{e-}4$), 177 (`G1`: overlap == 0.0), 178–179 (`G2`: persistence_off / median_narrow_e >= 100.0 and median_narrow_e <= 0.1 * persistence_on), 180–181 (`G4_below` >= 1.19e-5), 182 (`P6` <= 1.573e-3), 183 (`ARCH-PARITY` <= ARCH_PARITY_LIMIT), 190–191 (`blocking_failures`), 231–234 (`P3_broadening_repairs`: `np.median(broad_R) <= 0.25 * np.median(narrow_R)`), 240–243 (`G4_above` < persistence_off). |
| **2 concurrent fits, nice 10** | **MATCH** | `launch_driver.sh` lines 12–16 (`while [ "$(jobs -rp \| wc -l \| tr -d ' ')" -ge 2 ]; do wait -n; done; nohup nice -n 10 "$PY" run_family4b.py train "$seed"`). `run_family4b.py` lines 116–153 defines single-seed fit `train(seed)`. |

## Summary
- One-line fix applied cleanly to `run_family4b.py`.
- Import check passed without error.
- All 5 items from `TASK.md` verified line-by-line as **MATCH**.
- Directories `family4_second_architecture/`, `family1e_state_broadening_new_pairs/`, `recovery/`, and `pkg_scratch/` were not modified.
- No `smoke`, `train`, `gates`, or `evaluate` runs were launched.
