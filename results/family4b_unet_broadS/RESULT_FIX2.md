# Result of Fix 2 for Family 4b (`gates()` KeyError)

## 1. Problem Diagnosis

In `run_family4b.py`, `gates()` read `old = json.loads(F4_RESULT.read_text())` where `F4_RESULT` pointed to `pde_volume/family4/results/P1_diffusion_fisher_kpp/RESULT.json`.
Inspection of `RESULT.json` revealed top-level keys:
```json
["pair", "per_seed", "narrow", "broad", "P3_broadening_repairs"]
```
It did not contain a `"geometry"` key, resulting in `KeyError: 'geometry'` at line 172.

Inspection of Family 4's runner (`family4_second_architecture/run_family4.py`) showed:
- Geometry is computed by `geometry(pair)` during Family 4's gating step.
- The resulting dictionary is saved to `(RESULTS / pair / "GATES.json")` (i.e. `pde_volume/family4/results/P1_diffusion_fisher_kpp/GATES.json`).
- `GATES.json` records the top-level `"geometry"` object:
  ```json
  "geometry": {
    "training_support": [0.4865763783454895, 0.5169978737831116],
    "switch_support": [0.690571665763855, 0.7228957414627075],
    "overlap": 0.0
  }
  ```

## 2. Applied Minimal Diff

In `run_family4b.py`, replaced `geometry = old["geometry"]` with reading from `F4_RESULT.with_name("GATES.json")` if present, with fallback to `f4.geometry(PAIR)`:

```diff
--- a/run_family4b.py
+++ b/run_family4b.py
@@ -171,3 +171,6 @@
     persistence_off = float(np.median(_relative(np.asarray(sw), np.asarray(sw_target))))
-    geometry = old["geometry"]
+    f4_gates = F4_RESULT.with_name("GATES.json")
+    geometry = (old.get("geometry")
+                or (json.loads(f4_gates.read_text())["geometry"]
+                    if f4_gates.exists() else f4.geometry(PAIR)))
     payload = {
```

No gate thresholds, seeds, or evaluation logic were modified.
Protected directories (`family4_second_architecture/`, `family1e_*`, `recovery/`, `pkg_scratch/`) were untouched.

## 3. Verification Command & Verbatim Output

Command executed:
```sh
source pde_volume/ENV.sh; JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 pde_restart/env/bin/python run_family4b.py gates
```

Verbatim stdout:
```json
{
  "schema": "family4b-broadS-gates-v1",
  "written_before_any_off_support_number_was_read": true,
  "pair": "P1_diffusion_fisher_kpp",
  "sampler": "family1e.BROADS_BLOCKS",
  "geometry": {
    "training_support": [
      0.4865763783454895,
      0.5169978737831116
    ],
    "switch_support": [
      0.690571665763855,
      0.7228957414627075
    ],
    "overlap": 0.0
  },
  "G1": true,
  "G2": true,
  "G4_below": true,
  "P6": true,
  "ARCH-PARITY": true,
  "persistence_on": 0.12143667042255402,
  "persistence_off": 0.0801907330751419,
  "median_e_on_narrow_reused": 0.0005066899757366627,
  "narrow_R_reused_from_family4": [
    16.071811331697297,
    9.277517361090826,
    10.923484925069193,
    11.298444542335417,
    13.410201338366361,
    11.987632846513069,
    8.6486132652304,
    8.612029204208264,
    10.121310933426576,
    9.19035606483021
  ],
  "per_seed_e_on_broadS": {
    "0": 0.000570172443985939,
    "1": 0.0006199831841513515,
    "2": 0.0006750111933797598,
    "3": 0.0005641347379423678,
    "4": 0.0006036142003722489,
    "5": 0.0006504050688818097,
    "6": 0.0005495575605891645,
    "7": 0.0006738085066899657,
    "8": 0.0007332435343414545,
    "9": 0.0006089485250413418
  },
  "blocking_failures": []
}
```

Gate execution passed with `blocking_failures: []`. Gates payload was written to `pde_volume/family4b/results/GATES.json`.
`evaluate` was NOT run.
