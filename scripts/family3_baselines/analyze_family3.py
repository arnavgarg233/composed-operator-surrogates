#!/usr/bin/env python3
"""Family 3: order-blind and zero-contrast baselines for the Route E analysis.

READ-ONLY with respect to dependency/**. Writes only into
results/family3_baselines/.

Definitions, gates and tolerances are fixed in PREDECLARED.md, written before this
script was run. Nothing here may change a threshold.

Metric (path_vs_exposure.py:115,129,152-155):
    S_{i,W} = sqrt(mean_{j in W} q2[i,j]) / sqrt(mean_{j in W} b2[i,j])
Observables (s_of_t.py:249-257): trajectory = all 201 frames (= PVE window 'full');
endpoint = frame 200 (= PVE window 'endpoint').

Both new baselines emit a raw order contrast of exactly zero, so their centered
contrast prediction is -m and their residual against truth is -delta_i; hence
q2_baseline[i,j] = delta2[i,j], whose square root ships as
truth_scales.raw_contrast_rms_per_unit_frame. No truth FIELD is required, and none is
available: EVALUATOR_BUNDLE.npz is not shipped.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
PKG = RELEASE_ROOT / "dependency"
TAB = PKG / 'results/tables/route_e'
OUT = RELEASE_ROOT / "results" / "family3_baselines"

N_UNITS = 48
N_FRAMES = 201
BOOTSTRAP = 9999            # PVE:205
RNG_SEED = 20260909         # PVE:205
TOL = 1e-9                  # PREDECLARED.md section 5
SEEDS = ['0', '1', '2', '3', '4']
SENSITIVITY_SEEDS = ['0', '1', '2', '4']   # seed 3 dropped; SENSITIVITY only

# ----------------------------------------------------------------------------- utils

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def ratio(num, den):
    """path_vs_exposure/s_of_t ratio(): NaN where denominator is not > 0 (s_of_t.py:99-102)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    shape = np.broadcast_shapes(num.shape, den.shape)
    return np.divide(num, den, out=np.full(shape, np.nan), where=den > 0)


WINDOW = {'trajectory': slice(0, 201), 'endpoint': slice(200, 201)}


def window_S(q2: np.ndarray, b2: np.ndarray, obs: str) -> np.ndarray:
    w = WINDOW[obs]
    return ratio(np.sqrt(q2[:, w].mean(axis=1)), np.sqrt(b2[:, w].mean(axis=1)))


def summarize(values: np.ndarray, weights: np.ndarray) -> dict:
    """path_vs_exposure.py:100-109."""
    values = arr(values)
    valid = np.isfinite(values)
    usable = bool(valid.all())
    sampled = weights @ values if usable else None
    return {'mean': float(values.mean()),
            'median': float(np.median(values)),
            'total_units': int(values.size),
            'valid_units': int(valid.sum()),
            'status': 'ok' if usable else 'undefined',
            'ci95': [float(v) for v in np.quantile(sampled, [0.025, 0.975])] if usable else None,
            'ci90': [float(v) for v in np.quantile(sampled, [0.05, 0.95])] if usable else None}


def maxabs(a, b) -> float:
    a, b = arr(a), arr(b)
    d = np.abs(a - b)
    return float(np.nanmax(d)) if d.size else 0.0


# ------------------------------------------------------------------------ load inputs

print('loading shipped tables ...', flush=True)
pve_path, sot_path, re_path = (TAB / 'PATH_VS_EXPOSURE_RESULT.json',
                               TAB / 'S_OF_T_RESULT.json',
                               TAB / 'ROUTE_E_RESULT.json')
with pve_path.open() as fh:
    PVE = json.load(fh)
with sot_path.open() as fh:
    SOT = json.load(fh)
with re_path.open() as fh:
    ROUTE_E = json.load(fh)

assert PVE['shape'] == [5, 48, 201, 256] and SOT['shape'] == [5, 48, 201, 256]
assert PVE['uncertainty']['bootstrap_replicates'] == BOOTSTRAP
assert PVE['uncertainty']['rng_seed'] == RNG_SEED
assert PVE['uncertainty']['independent_units'] == N_UNITS

# Truth NORMS (not fields).
b_rms_pve = arr(PVE['truth_scales']['centered_rms_per_unit_frame'])      # PVE:263 sqrt(b2)
delta_rms = arr(PVE['truth_scales']['raw_contrast_rms_per_unit_frame'])  # PVE:264 sqrt(delta2)
b_rms_sot = arr(SOT['units']['b'])                                       # SOT:288 sqrt(b2)
assert b_rms_pve.shape == delta_rms.shape == b_rms_sot.shape == (N_UNITS, N_FRAMES)
b2 = b_rms_pve ** 2
delta2 = delta_rms ** 2
truth_scale_agreement = maxabs(b_rms_pve, b_rms_sot)

# Learner residual energies q2[i,j] per record, from sqrt(q2) shipped at s_of_t.py:241.
q2 = {'ensemble': arr(SOT['ensemble']['units']['q']) ** 2,
      'seed_error_pool': arr(SOT['seed_error_pool']['units']['q']) ** 2}
for s in SEEDS:
    q2['seed' + s] = arr(SOT['per_seed'][s]['units']['q']) ** 2
for k, v in q2.items():
    assert v.shape == (N_UNITS, N_FRAMES), (k, v.shape)

# Bootstrap weights: identical construction to path_vs_exposure.py:226.
weights = np.random.default_rng(RNG_SEED).multinomial(
    N_UNITS, np.full(N_UNITS, 1 / N_UNITS), BOOTSTRAP) / N_UNITS

# --------------------------------------------------------------- V1/V2/V3/V4/V5 checks

print('validating metric implementation against shipped numbers ...', flush=True)
PVE_WIN = {'trajectory': 'full', 'endpoint': 'endpoint'}
validation = {'tolerance': TOL, 'checks': {}}

# Recomputed per-unit S for every record that has both q2 (SOT) and shipped S (PVE).
S_recomputed, rawS_recomputed = {}, {}
for rec, pve_key in (('ensemble', 'ensemble'), *((f'seed{s}', ('per_seed', s)) for s in SEEDS)):
    node = PVE[pve_key] if isinstance(pve_key, str) else PVE[pve_key[0]][pve_key[1]]
    for obs, wname in PVE_WIN.items():
        S_recomputed[(rec, obs)] = window_S(q2[rec], b2, obs)
        rawS_recomputed[(rec, obs)] = window_S(q2[rec], delta2, obs)

# PVE 'pooled' = mean over seeds of per-seed per-unit S (path_vs_exposure.py:154 + :249).
for obs in WINDOW:
    S_recomputed[('pooled', obs)] = np.mean([S_recomputed[(f'seed{s}', obs)] for s in SEEDS], axis=0)
    rawS_recomputed[('pooled', obs)] = np.mean([rawS_recomputed[(f'seed{s}', obs)] for s in SEEDS], axis=0)
# SOT 'seed_error_pool' = mean over seeds of q2, then the ratio (s_of_t.py:305-306).
for obs in WINDOW:
    S_recomputed[('seed_error_pool', obs)] = window_S(q2['seed_error_pool'], b2, obs)
    rawS_recomputed[('seed_error_pool', obs)] = window_S(q2['seed_error_pool'], delta2, obs)

v1, v2 = {}, {}
for rec, pve_key in (('ensemble', 'ensemble'), ('pooled', 'pooled'),
                     *((f'seed{s}', ('per_seed', s)) for s in SEEDS)):
    node = PVE[pve_key] if isinstance(pve_key, str) else PVE[pve_key[0]][pve_key[1]]
    for obs, wname in PVE_WIN.items():
        shipped = arr(node['windows'][wname]['contrast']['S']['per_unit'])
        shipped_raw = arr(node['windows'][wname]['contrast']['raw_S']['per_unit'])
        v1[f'{rec}.{obs}'] = maxabs(S_recomputed[(rec, obs)], shipped)
        v2[f'{rec}.{obs}'] = maxabs(rawS_recomputed[(rec, obs)], shipped_raw)
validation['checks']['V1_S_per_unit_vs_PATH_VS_EXPOSURE'] = {
    'max_abs_diff': max(v1.values()), 'per_record': v1, 'pass': max(v1.values()) <= TOL}
validation['checks']['V2_rawS_per_unit_vs_PATH_VS_EXPOSURE'] = {
    'max_abs_diff': max(v2.values()), 'per_record': v2, 'pass': max(v2.values()) <= TOL}

# V3: recognition-only S must be 1 per unit (path_vs_exposure.py:228-231).
v3 = {}
for obs, wname in PVE_WIN.items():
    s = arr(PVE['recognition_only']['windows'][wname]['contrast']['S']['per_unit'])
    v3[obs] = maxabs(s, np.ones_like(s))
validation['checks']['V3_recognition_only_S_equals_one'] = {
    'max_abs_diff': max(v3.values()), 'per_observable': v3, 'pass': max(v3.values()) <= TOL}

# V4: bootstrap CI reproduction (same RNG stream, same weights).
v4 = {}
for rec, pve_key in (('ensemble', 'ensemble'), ('pooled', 'pooled'),
                     *((f'seed{s}', ('per_seed', s)) for s in SEEDS)):
    node = PVE[pve_key] if isinstance(pve_key, str) else PVE[pve_key[0]][pve_key[1]]
    for obs, wname in PVE_WIN.items():
        w = node['windows'][wname]
        mine = summarize(arr(w['contrast']['S']['per_unit']), weights)['ci95']
        v4[f'{rec}.{obs}.S'] = maxabs(mine, w['contrast']['S']['ci95'])
        mmine = summarize(arr(w['margins_vs_recognition_only']['S']['per_unit']), weights)['ci95']
        v4[f'{rec}.{obs}.margin_vs_recognition_only'] = maxabs(
            mmine, w['margins_vs_recognition_only']['S']['ci95'])
validation['checks']['V4_bootstrap_ci95_reproduction'] = {
    'max_abs_diff': max(v4.values()), 'detail': v4, 'pass': max(v4.values()) <= TOL,
    'note': 'same construction as path_vs_exposure.py:226 and :104,:108'}

# V5: S = raw_S * S_zero identity, cross-checking truth_scales against shipped ratios only.
S_zero = {obs: window_S(delta2, b2, obs) for obs in WINDOW}   # q2 of the new baselines = delta2
v5 = {}
for rec in ('ensemble', 'pooled', *(f'seed{s}' for s in SEEDS)):
    for obs in WINDOW:
        if rec == 'pooled':
            continue  # 'pooled' averages ratios across seeds; the identity is per-seed
        implied = ratio(S_recomputed[(rec, obs)], rawS_recomputed[(rec, obs)])
        v5[f'{rec}.{obs}'] = maxabs(implied, S_zero[obs])
validation['checks']['V5_S_equals_rawS_times_Szero'] = {
    'max_abs_diff': max(v5.values()), 'detail': v5, 'pass': max(v5.values()) <= TOL}
validation['checks']['truth_scale_cross_file_agreement'] = {
    'max_abs_diff': truth_scale_agreement,
    'note': 'sqrt(b2) from PATH_VS_EXPOSURE truth_scales vs S_OF_T units.b',
    'pass': truth_scale_agreement <= TOL}

validation['all_gating_checks_pass'] = all(
    validation['checks'][k]['pass'] for k in
    ('V1_S_per_unit_vs_PATH_VS_EXPOSURE', 'V2_rawS_per_unit_vs_PATH_VS_EXPOSURE',
     'V3_recognition_only_S_equals_one', 'V5_S_equals_rawS_times_Szero'))
validation['V4_ci_pass'] = validation['checks']['V4_bootstrap_ci95_reproduction']['pass']

# --------------------------------------------------- extra (NOT predeclared, not a gate)

print('additional npz consistency check (not a gate) ...', flush=True)
npz_check = {'status': 'not run'}
try:
    raw_pred_contrast2 = []
    npz_receipts = []
    per_seed_order_sensitivity = {}
    for s in SEEDS:
        p = PKG / f'data/learner_artifacts/composed_seed{s}.npz'
        with np.load(p, allow_pickle=False) as z:
            pa = np.asarray(z['pred_ab'], dtype=np.float64)
            pb = np.asarray(z['pred_ba'], dtype=np.float64)
        assert pa.shape == pb.shape == (48, 201, 256)
        assert np.array_equal(pa[:, 0], pb[:, 0]), 'initial states differ'
        diff = pa - pb
        raw_pred_contrast2.append(np.mean(diff ** 2, axis=-1))
        per_seed_order_sensitivity[s] = {
            'pred_ab_and_pred_ba_bit_identical': bool(np.array_equal(pa, pb)),
            'max_abs_raw_predicted_contrast': float(np.abs(diff).max()),
            'rms_raw_predicted_contrast_all_frames': float(np.sqrt(np.mean(diff ** 2))),
            'rms_raw_predicted_contrast_endpoint': float(np.sqrt(np.mean(diff[:, 200] ** 2)))}
        npz_receipts.append({'path': str(p), 'sha256': sha256(p)})
        del pa, pb, diff
    # Triangle bound linking npz predictions to shipped q2 and delta2, per seed:
    bounds_ok = True
    worst = 0.0
    for idx, s in enumerate(SEEDS):
        P = np.sqrt(raw_pred_contrast2[idx])            # || pred contrast ||  (from npz)
        D = delta_rms                                    # || delta ||          (shipped truth norm)
        Q = np.sqrt(q2['seed' + s])                      # || pred contrast - delta || (shipped)
        lo, hi = np.abs(P - D), P + D
        slack = np.maximum(lo - Q, Q - hi)
        worst = max(worst, float(np.max(slack)))
        bounds_ok = bounds_ok and bool(np.all(slack <= 1e-9))
    npz_check = {
        'status': 'run',
        'shapes_ok': True,
        'initial_states_equal_across_arms': True,
        'order_blind_raw_contrast_is_exactly_zero': True,
        'triangle_bound_|P-D| <= Q <= P+D_holds_all_seeds_units_frames': bounds_ok,
        'worst_violation': worst,
        'meaning': ('links the shipped residual norms q2 and the shipped truth contrast '
                    'norms delta2 to the actual npz predictions; NOT a predeclared gate'),
        'per_seed_order_sensitivity': per_seed_order_sensitivity,
        'seeds_that_are_exactly_order_blind': [
            s for s, v in per_seed_order_sensitivity.items()
            if v['pred_ab_and_pred_ba_bit_identical']],
        'finding': ('seed 3 emits bit-identical pred_ab and pred_ba: it IS the order-blind '
                    'predictor, not merely close to it. Its raw_S is exactly 1 and its margin '
                    'against the order-blind / zero-contrast baseline is exactly 0 with a '
                    'degenerate CI [0,0]. This was discovered after the gate was fixed and '
                    'changes no threshold.'),
        'npz_receipts': npz_receipts}
    del raw_pred_contrast2
except Exception as exc:  # pragma: no cover
    npz_check = {'status': 'failed', 'error': repr(exc)}

# ------------------------------------------------------------------------- baselines

print('computing baselines and margins ...', flush=True)

# Per-unit S for each baseline, per observable.
baseline_S = {}
for obs, wname in PVE_WIN.items():
    ones = np.ones(N_UNITS)
    baseline_S[('common_pattern', obs)] = ones.copy()
    baseline_S[('recognition_only', obs)] = arr(
        PVE['recognition_only']['windows'][wname]['contrast']['S']['per_unit'])
    baseline_S[('order_blind', obs)] = S_zero[obs].copy()
    baseline_S[('zero_contrast', obs)] = S_zero[obs].copy()
    baseline_S[('first_leg_oracle', obs)] = arr(
        PVE['first_leg_oracle']['windows'][wname]['contrast']['S']['per_unit'])

identity_orderblind_zero = max(
    maxabs(baseline_S[('order_blind', o)], baseline_S[('zero_contrast', o)]) for o in WINDOW)

GATED = ('common_pattern', 'recognition_only', 'order_blind', 'zero_contrast')
CONTEXT = ('first_leg_oracle',)

RECORDS = [('ensemble', 'PRIMARY'), ('pooled', 'secondary'), ('seed_error_pool', 'secondary'),
           *[(f'seed{s}', 'secondary') for s in SEEDS]]

results = {}
for rec, role in RECORDS:
    entry = {'role': role, 'observables': {}}
    for obs in WINDOW:
        s_learner = S_recomputed[(rec, obs)]
        item = {'S_learner': summarize(s_learner, weights),
                'raw_S_learner_mean': float(np.mean(rawS_recomputed[(rec, obs)])),
                'baselines': {}}
        for bname in GATED + CONTEXT:
            sb = baseline_S[(bname, obs)]
            margin = s_learner - sb
            msum = summarize(margin, weights)
            ci = msum['ci95']
            item['baselines'][bname] = {
                'S_baseline': summarize(sb, weights),
                'margin_S_learner_minus_baseline': msum,
                'ci95_entirely_below_zero': bool(ci is not None and ci[1] < 0.0),
                'gated': bname in GATED}
        entry['observables'][obs] = item
    results[rec] = entry

# ------------------------------------------------------------------------------ gate

primary = results['ensemble']['observables']['trajectory']['baselines']
gate_rows = {b: primary[b]['ci95_entirely_below_zero'] for b in GATED}
gate_pass = all(gate_rows.values())
endpoint_primary = results['ensemble']['observables']['endpoint']['baselines']
gate = {
    'definition': ('PREDECLARED.md section 4: PASS iff, on the trajectory observable for the '
                   'PRIMARY record (ensemble), the 95% CI of the paired margin '
                   'S_learner - S_baseline lies entirely below 0 against all four fair '
                   'baselines. Endpoint reported, not gated.'),
    'record': 'ensemble', 'observable': 'trajectory',
    'per_baseline_ci95_below_zero': gate_rows,
    'verdict': 'PASS' if gate_pass else 'FAIL',
    'endpoint_reported_not_gated': {
        b: {'margin_mean': endpoint_primary[b]['margin_S_learner_minus_baseline']['mean'],
            'margin_ci95': endpoint_primary[b]['margin_S_learner_minus_baseline']['ci95'],
            'ci95_entirely_below_zero': endpoint_primary[b]['ci95_entirely_below_zero']}
        for b in GATED}}

# ----------------------------------------------------------------------- sensitivity

print('seed-3 sensitivity ...', flush=True)
sensitivity = {
    'label': 'SENSITIVITY ONLY - not primary, does not replace the five-seed result',
    'reason': ROUTE_E['divergence_rule'],
    'seed3_primitive_recovery': ROUTE_E['composed_per_seed']['3'],
    'five_seed_median_primitive_errors': ROUTE_E['median'],
    'ensemble_without_seed3': {
        'status': 'NOT COMPUTABLE from shipped files',
        'reason': ('the four-seed ensemble residual needs the cross-seed spatial inner '
                   'products mean_x(res_k . res_l), which require the truth field delta_i; '
                   'EVALUATOR_BUNDLE.npz is not shipped. Only per-seed-averaged records '
                   '(pooled, seed_error_pool) can be recomputed on a seed subset.')},
    'records': {}}
for label, seed_set in (('all_seeds', SEEDS), ('without_seed3', SENSITIVITY_SEEDS)):
    for rec_kind in ('pooled', 'seed_error_pool'):
        block = {}
        for obs in WINDOW:
            if rec_kind == 'pooled':
                s_learner = np.mean([S_recomputed[(f'seed{s}', obs)] for s in seed_set], axis=0)
            else:
                pooled_q2 = np.mean([q2['seed' + s] for s in seed_set], axis=0)
                s_learner = window_S(pooled_q2, b2, obs)
            obs_block = {'S_learner': summarize(s_learner, weights), 'baselines': {}}
            for bname in GATED:
                margin = s_learner - baseline_S[(bname, obs)]
                msum = summarize(margin, weights)
                obs_block['baselines'][bname] = {
                    'margin_mean': msum['mean'], 'margin_ci95': msum['ci95'],
                    'ci95_entirely_below_zero': bool(msum['ci95'][1] < 0.0)}
            block[obs] = obs_block
        sensitivity['records'][f'{rec_kind}__{label}'] = block
sensitivity['gate_would_flip'] = {
    rec_kind: (all(sensitivity['records'][f'{rec_kind}__all_seeds']['trajectory']
                   ['baselines'][b]['ci95_entirely_below_zero'] for b in GATED)
               != all(sensitivity['records'][f'{rec_kind}__without_seed3']['trajectory']
                      ['baselines'][b]['ci95_entirely_below_zero'] for b in GATED))
    for rec_kind in ('pooled', 'seed_error_pool')}

# ---------------------------------------------------------------------------- output

truth = {
    'evaluator_bundle_shipped': False,
    'truth_fields_recoverable': False,
    'why': ('data/corpus/EVALUATOR_BUNDLE.npz is absent (data/manifest.json not_shipped; '
            'CLAIM_INVENTORY sections 1 and 2 row 14). The truth fields y^AB, y^BA, delta_i, '
            'd_i and the training-mean contrast m are therefore unavailable at spatial '
            'resolution: 5 shipped per-frame scalars per unit cannot determine 256 unknowns.'),
    'truth_quantities_that_ARE_shipped': {
        'per_unit_per_frame_sqrt_b2 (centered truth contrast RMS, ||d_i||)':
            'PATH_VS_EXPOSURE_RESULT.json truth_scales.centered_rms_per_unit_frame '
            '(= S_OF_T_RESULT.json units.b)',
        'per_unit_per_frame_sqrt_delta2 (raw truth contrast RMS, ||delta_i||)':
            'PATH_VS_EXPOSURE_RESULT.json truth_scales.raw_contrast_rms_per_unit_frame',
        'per_unit_per_frame_sqrt_y2 (||y^AB||)': 'S_OF_T_RESULT.json units.Y',
        'per_unit_per_frame_sqrt_q2 (learner residual norm)':
            'S_OF_T_RESULT.json <record>.units.q',
        'm (training-mean contrast) as a FIELD': 'NOT shipped',
        'W (learner weights)': 'data/learner_artifacts/weights_seed{0..4}.pt shipped but '
                               'never loaded here; the analysis needs no weights'},
    'baselines_computable_from_shipped_data_alone': True,
    'baselines_computable_why': (
        'both new baselines emit a raw order contrast of exactly zero, so their residual '
        'against truth is -delta_i and their q2 equals delta2, whose square root is shipped. '
        'No truth field is needed.'),
    'not_computed_because_truth_absent': [
        'arm-level (per-ordering) relative L2 and RMSE of the order-blind predictor',
        'a truth-side order-blind reference as a distinct number (identical to the above on '
        'the contrast metric S; its arm-level metrics need the truth fields)',
        'any four-seed ensemble record'],
    'forbidden_substitution_not_made': (
        'no baseline was scored against the learner predictions in place of truth')}

payload = {
    'schema': 'pde-family3-baselines-v1',
    'generated': '2026-09-10',
    'predeclaration': str(OUT / 'PREDECLARED.md'),
    'package': str(PKG),
    'inputs': {'PATH_VS_EXPOSURE_RESULT.json': {'path': str(pve_path), 'sha256': sha256(pve_path)},
               'S_OF_T_RESULT.json': {'path': str(sot_path), 'sha256': sha256(sot_path)},
               'ROUTE_E_RESULT.json': {'path': str(re_path), 'sha256': sha256(re_path)}},
    'truth_availability': truth,
    'metric': {'S': 'sqrt(mean_W q2) / sqrt(mean_W b2), per unit, then mean over 48 units',
               'source_lines': 'path_vs_exposure.py:115,129,152-155; s_of_t.py:249-257',
               'observables': {'trajectory': 'all 201 frames (PVE window full = [0,200])',
                               'endpoint': 'frame 200 only'}},
    'uncertainty': {'independent_units': N_UNITS, 'bootstrap_replicates': BOOTSTRAP,
                    'rng_seed': RNG_SEED, 'resampling': 'paired whole units',
                    'construction': 'path_vs_exposure.py:226',
                    'seed_axis_is_not_independent_units': True},
    'validation': validation,
    'additional_npz_consistency_check_not_a_gate': npz_check,
    'baseline_identity': {
        'order_blind_equals_zero_contrast_max_abs_diff': identity_orderblind_zero,
        'note': ('predeclared in PREDECLARED.md 1(a): any predictor emitting one field for '
                 'both orderings has zero raw contrast, so it scores identically to the '
                 'zero-contrast control on S. Two motivations, one number.')},
    'baseline_S_per_unit_mean': {
        f'{b}.{o}': float(np.mean(baseline_S[(b, o)])) for b in GATED + CONTEXT for o in WINDOW},
    'results': results,
    'gate': gate,
    'sensitivity_seed3': sensitivity,
    'environment': {'numpy': np.__version__,
                    'python': __import__('sys').version.split()[0]},
    'inputs_modified': 'none; recovery/ opened read-only'}


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


(OUT / 'RESULT.json').write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + '\n')
print('wrote', OUT / 'RESULT.json')
print('validation pass:', validation['all_gating_checks_pass'], '| V4 CI pass:', validation['V4_ci_pass'])
print('GATE:', gate['verdict'], gate_rows)
for obs in ('trajectory', 'endpoint'):
    o = results['ensemble']['observables'][obs]
    print(f"  ensemble {obs}: S_learner={o['S_learner']['mean']:.6f}")
    for b in GATED:
        m = o['baselines'][b]['margin_S_learner_minus_baseline']
        print(f"    vs {b:18s} S_base={o['baselines'][b]['S_baseline']['mean']:.6f} "
              f"margin={m['mean']:+.6f} CI=[{m['ci95'][0]:+.6f},{m['ci95'][1]:+.6f}] "
              f"{'BELOW0' if o['baselines'][b]['ci95_entirely_below_zero'] else 'NOT-BELOW0'}")
