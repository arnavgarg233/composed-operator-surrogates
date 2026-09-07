"""Blind Route E diagnostic: path recognition versus equal-exposure composition.

Notation follows the paper, sections 2.1, 2.2, 2.7, 3.8, 3.9, and P2 section
2.2. We write y_i^AB(t,x)=B_(t-1) A_1 u_i for 1<=t<=2, with first leg A_t u_i;
y_i^BA reverses A and B. AB means A THEN B, not matrix multiplication order.
Frames j=0,...,200 have t_j=j/100, and frame 100 is the first-leg endpoint.
The model carries its predicted state across that handoff, without a reset.

EXPOSURE: cumulative (A-time,B-time) is (min(t,1),max(t-1,0)) for AB and its
reverse for BA. Cross-order exposure is equal only at frames 0 and 200. Own-arm
errors below always match prediction to truth at the same schedule and time.
First-leg and second-leg errors are NOT an equal-exposure comparison across arms.
Only the nontrivial endpoint here estimates equal-exposure composition accuracy;
this capture cannot supply the family B_s A_s u versus A_s B_s u for varying s.

CONTRAST: delta_i=y_i^AB-y_i^BA; m=mean_train(delta_j); d_i=delta_i-m;
dhat_ki=pred_ki^AB-pred_ki^BA-m. We compute in float64 from saved arrays. For
observable/window W, S_ki,W=||dhat_ki-d_i||_W/||d_i||_W, averaging the unitwise
norm ratios to obtain Sbar_k,W. Norms sum uniformly over time and space. Smaller
S is better; skill=1-Sbar is higher-is-better, not the paper's original score.
The numerator is ||e^AB-e^BA||, not a single-arm error. raw_S replaces ||d_i||
by ||delta_i|| to display how much an uncentered metric rewards a shared pattern.
S_pooled=sqrt(mean_ki ||e^AB-e^BA||_W^2 / mean_i ||d_i||_W^2) is a distinct
ratio-of-pooled-norms diagnostic, never a substitute for Sbar.

ORDER RECOGNITION: for each k,i,j, raw_sign_agreement is the fraction of spatial
points with sign(delta_hat)=sign(delta) among |delta|>1e-12. Predictions within
+/-1e-12 are ties and count as failures. A zero eligible count is undefined,
not success. centered_sign_agreement uses dhat,d and the identical rule. We
also report raw_direction_agreement=1[<delta_hat,delta>_x>0] at each unit/frame
with ||delta||_x>1e-12. This tests a vector direction, not amplitude accuracy.
Window sign fractions pool eligible point counts within each unit, then average
units; window direction fractions average eligible frames within each unit.
No frame or spatial coordinate is treated as an independent observation.

STATE ACCURACY: e^o_ki=pred^o_ki-y^o_i and R^o_ki,W=||e^o_ki||_W/||y^o_i||_W
for o=AB,BA. RMSE^o_ki,W=sqrt(mean_(t,x in W) (e^o_ki)^2). We keep both
orderings separate, return per-unit values and their mean/median, and also
pooled R=sqrt(mean_ki ||e^o_ki||_W^2 / mean_i ||y^o_i||_W^2). Framewise truth
scales are sqrt(mean_x d_i^2) and sqrt(mean_x delta_i^2). These are end-to-end
rollout errors, not isolated downstream operator errors.

BASELINES: mu^o=mean_evaluator_train y^o. The recognition-only predictor is
mu^AB for AB and mu^BA for BA at every frame, including frame 0. It knows order
but no held-out unit. Its contrast is m, so centered S=1 on every nondegenerate
window, up to arithmetic. We never fit the baseline on test units. Evaluator
training compositions are a diagnostic privilege, not learner training data.
The extra first_leg_oracle uses each test truth through frame 100 and mu^o on
frames 101:200. It is a nondeployable attribution control, not a fair predictor:
its full S_i^2 equals the post-handoff fraction of ||d_i||^2. It asks how much
full-path skill could arise from perfect first-leg prediction alone.

WINDOWS: full=0:200, first_leg=0:100, second_leg=100:200, post_handoff=101:200,
endpoint=200, all inclusive. Frame 100 belongs to both named legs but is counted
only once in full. Energy partitions use first_leg plus post_handoff, which are
disjoint. Signal and residual fractions sum squared d and e^AB-e^BA over those
pieces. They decompose squared pooled norms, NOT the mean of per-unit ratios.

POOLING: per_seed reports each frozen model. pooled averages per-seed metrics
within each matched unit, then averages units, with equal seed weights. Its
pooled norm diagnostics instead pool error energies. ensemble averages predicted
fields before computing anything, matching train_eval_v2:330-339. These objects
are not interchangeable. We do not select seeds. CIs are percentile intervals
from 9,999 paired whole-unit resamples shared across seeds, arms, and windows,
RNG 20260909. They are conditional on the fitted models, training means, and
fixed demonstration policy, not estimates of population training-seed variance.
Margins subtract the recognition-only metric on the SAME unit; the separately
named oracle margin instead subtracts its unit score. Negative favors the learner.
Degenerate ratio denominators propagate undefined summaries and intervals rather
than silently dropping units. Sign eligibility is explicit.

CLI real-input use requires terminal, frozen inputs and an adopted reading rule;
--confirm-frozen-inputs is an operator attestation, not a job-status check. No
weights, training modules, GPU, or network are used. --selftest reads no real
predictions or corpus and tests only synthetic arrays and a temporary synthetic
capture under this directory. --selftest --mutation swap-arms must exit nonzero.
We label synthetic measurements as synthetic; no scientific verdict is assigned.
"""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from s_of_t import json_ready, load_predictions, load_truth, numeric, ratio, receipt, require

WINDOWS = {'full': (0, 200), 'first_leg': (0, 100), 'second_leg': (100, 200),
           'post_handoff': (101, 200), 'endpoint': (200, 200)}
SIGN_TOLERANCE = 1e-12


def summary(values, weights):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    usable = bool(valid.all())
    sampled = weights @ values if usable else None
    return {'per_unit': values, 'mean': float(values.mean()),
            'median': float(np.median(values)), 'total_units': len(values),
            'valid_units': int(valid.sum()), 'status': 'ok' if usable else 'undefined',
            'ci95': np.quantile(sampled, [0.025, 0.975]) if usable else None,
            'ci90': np.quantile(sampled, [0.05, 0.95]) if usable else None}


def moments(pa, pb, a, b, pattern):
    ea, eb = pa - a, pb - b
    delta, delta_hat = a - b, pa - pb
    result = {'q2': np.mean((ea - eb)**2, axis=-1),
              'ab2': np.mean(ea**2, axis=-1), 'ba2': np.mean(eb**2, axis=-1)}
    for name, true, pred in (('raw', delta, delta_hat),
                             ('centered', delta - pattern, delta_hat - pattern)):
        eligible = np.abs(true) > SIGN_TOLERANCE
        hits = eligible & (np.abs(pred) > SIGN_TOLERANCE) & (np.sign(pred) == np.sign(true))
        result[name + '_hits'] = hits.sum(axis=-1)
    result['direction_hits'] = (np.sum(delta_hat * delta, axis=-1) > 0).astype(float)
    require(all(np.isfinite(v).all() for v in result.values()), 'prediction moment overflow')
    return result


def truth_moments(a, b, pattern):
    delta, d = a - b, a - b - pattern
    result = {'b2': np.mean(d**2, axis=-1), 'delta2': np.mean(delta**2, axis=-1),
              'yab2': np.mean(a**2, axis=-1), 'yba2': np.mean(b**2, axis=-1),
              'raw_count': (np.abs(delta) > SIGN_TOLERANCE).sum(axis=-1),
              'centered_count': (np.abs(d) > SIGN_TOLERANCE).sum(axis=-1),
              'direction_count': (np.linalg.norm(delta, axis=-1) > SIGN_TOLERANCE).astype(int)}
    require(all(np.isfinite(v).all() for v in result.values()), 'truth moment overflow')
    return result


def block(items, truth, weights):
    m = {key: np.stack([item[key] for item in items]) for key in items[0]}
    score_frame = ratio(np.sqrt(m['q2']), np.sqrt(truth['b2']))
    result = {'windows': {}, 'framewise': {
        'S_per_unit': score_frame.mean(axis=0),
        'S_mean': score_frame.mean(axis=(0, 1)),
        'raw_eligible_points': truth['raw_count'],
        'centered_eligible_points': truth['centered_count'],
        'raw_sign_agreement': ratio(m['raw_hits'].mean(axis=0), truth['raw_count']),
        'centered_sign_agreement': ratio(m['centered_hits'].mean(axis=0), truth['centered_count']),
        'raw_direction_agreement': ratio(m['direction_hits'].mean(axis=0), truth['direction_count']),
        'ab_relative_l2': ratio(np.sqrt(m['ab2']), np.sqrt(truth['yab2'])).mean(axis=0),
        'ba_relative_l2': ratio(np.sqrt(m['ba2']), np.sqrt(truth['yba2'])).mean(axis=0)}}
    for name, (start, stop) in WINDOWS.items():
        w = slice(start, stop + 1)
        q2, b2 = m['q2'][:, :, w].mean(axis=-1), truth['b2'][:, w].mean(axis=-1)
        s = ratio(np.sqrt(q2), np.sqrt(b2)).mean(axis=0)
        raw_s = ratio(np.sqrt(q2), np.sqrt(truth['delta2'][:, w].mean(axis=-1))).mean(axis=0)
        item = {'contrast': {'S': summary(s, weights), 'skill_mean': float(1 - s.mean()),
                             'S_pooled': float(ratio(np.sqrt(q2.mean()), np.sqrt(b2.mean()))),
                             'raw_S': summary(raw_s, weights)}, 'arms': {}, 'recognition': {}}
        for arm in ('ab', 'ba'):
            e2 = m[arm + '2'][:, :, w].mean(axis=-1)
            y2 = truth['y' + arm + '2'][:, w].mean(axis=-1)
            item['arms'][arm] = {
                'relative_l2': summary(ratio(np.sqrt(e2), np.sqrt(y2)).mean(axis=0), weights),
                'rmse': summary(np.sqrt(e2).mean(axis=0), weights),
                'pooled_relative_l2': float(ratio(np.sqrt(e2.mean()), np.sqrt(y2.mean())))}
        for mode in ('raw', 'centered', 'direction'):
            counts = truth[mode + '_count'][:, w]
            hits = m[mode + '_hits'][:, :, w]
            if mode == 'direction':
                hits = hits * counts
            fraction = ratio(hits.sum(axis=-1), counts.sum(axis=-1)).mean(axis=0)
            item['recognition'][mode] = summary(fraction, weights)
            item['recognition'][mode]['eligible_per_unit'] = counts.sum(axis=-1)
        result['windows'][name] = item
    result['energy_partition'] = {}
    for label, array in (('signal_fraction', truth['b2']), ('residual_fraction', m['q2'])):
        total = array.sum()
        result['energy_partition'][label] = {
            name: float(ratio(array[..., start:stop + 1].sum(), total))
            for name, (start, stop) in WINDOWS.items() if name in ('first_leg', 'post_handoff')}
    return result


def add_margins(record, baseline, weights):
    for name, window in record['windows'].items():
        reference = baseline['windows'][name]
        window['margins_vs_recognition_only'] = {
            'S': summary(window['contrast']['S']['per_unit'] - reference['contrast']['S']['per_unit'], weights),
            'arms': {arm: summary(window['arms'][arm]['relative_l2']['per_unit']
                                  - reference['arms'][arm]['relative_l2']['per_unit'], weights)
                     for arm in ('ab', 'ba')}}


def exposure():
    times = np.arange(201, dtype=float) / 100
    ab = np.column_stack([np.minimum(times, 1), np.maximum(times - 1, 0)])
    ba = ab[:, ::-1]
    return {'times': times, 'columns': ['A_time', 'B_time'], 'ab': ab, 'ba': ba,
            'equal_exposure_frames': np.flatnonzero(np.all(ab == ba, axis=1)).tolist(),
            'nontrivial_equal_exposure_frame': 200,
            'second_leg_is_equal_exposure_family': False}


def analyze(pred_ab, pred_ba, truth_ab, truth_ba, train_ab, train_ba,
            seeds=None, bootstrap=9999, rng_seed=20260909):
    pa, pb, a, b, ta, tb = [numeric(v, name) for v, name in zip(
        (pred_ab, pred_ba, truth_ab, truth_ba, train_ab, train_ba),
        ('pred_ab', 'pred_ba', 'truth_ab', 'truth_ba', 'train_ab', 'train_ba'))]
    require(pa.ndim == 4 and pa.shape == pb.shape, 'predictions need equal [seed, unit, frame, space] shapes')
    k, n, f, x = pa.shape
    require(k >= 1 and n >= 2 and f == 201 and x >= 1, 'need >=1 seed, >=2 units, 201 frames')
    require(a.shape == b.shape == pa.shape[1:], 'truth and prediction shapes differ')
    require(ta.ndim == 3 and ta.shape == tb.shape and ta.shape[1:] == (f, x)
            and len(ta) >= 1, 'training arms do not match')
    require(np.array_equal(a[:, 0], b[:, 0]) and np.array_equal(ta[:, 0], tb[:, 0]),
            'truth initial arms differ')
    require(np.array_equal(pa[:, :, 0], np.broadcast_to(a[:, 0], (k, n, x)))
            and np.array_equal(pb[:, :, 0], pa[:, :, 0]), 'prediction initial states differ')
    seeds = list(range(k)) if seeds is None else list(seeds)
    require(len(seeds) == k and len(set(seeds)) == k, 'unique seed IDs must match seed axis')
    require(bootstrap >= 100, 'at least 100 bootstrap replicates required')
    pattern = (ta - tb).mean(axis=0)
    ma, mb = ta.mean(axis=0), tb.mean(axis=0)
    require(np.allclose(ma - mb, pattern, rtol=1e-12, atol=1e-12), 'mean/contrast identity failed')
    truth = truth_moments(a, b, pattern)
    weights = np.random.default_rng(rng_seed).multinomial(n, np.full(n, 1 / n), bootstrap) / n
    baseline = block([moments(ma, mb, a, b, pattern)], truth, weights)
    for window in baseline['windows'].values():
        s = window['contrast']['S']['per_unit']
        require(np.allclose(s[np.isfinite(s)], 1, rtol=1e-10, atol=1e-10),
                'recognition-only centered S must equal one on nondegenerate units')
    result = {'schema': 'pde-path-vs-exposure-v1', 'model_seeds': seeds, 'shape': list(pa.shape),
              'windows': {name: list(bounds) for name, bounds in WINDOWS.items()},
              'exposure': exposure(), 'sign_tolerance': SIGN_TOLERANCE,
              'centering': 'same float64 evaluator-training contrast mean for truth and prediction',
              'score_direction': 'S lower is better; skill=1-S higher is better',
              'uncertainty': {'independent_units': n, 'bootstrap_replicates': bootstrap,
                              'rng_seed': rng_seed, 'resampling': 'paired whole units, shared across seeds/arms/windows',
                              'conditional_on': 'fixed training means, models and demonstrations',
                              'seed_axis_is_not_independent_units': True},
              'recognition_only': baseline, 'per_seed': {}}
    all_moments = []
    for index, seed in enumerate(seeds):
        item = moments(pa[index], pb[index], a, b, pattern)
        all_moments.append(item)
        record = block([item], truth, weights)
        add_margins(record, baseline, weights)
        result['per_seed'][str(seed)] = record
    result['pooled'] = block(all_moments, truth, weights)
    result['pooled']['estimand'] = 'equal-seed metric mean within unit, then equal-unit mean; pooled norms use energies'
    result['ensemble'] = block([moments(pa.mean(axis=0), pb.mean(axis=0), a, b, pattern)], truth, weights)
    result['ensemble']['estimand'] = 'mean predicted fields before scoring, as in train_eval_v2'
    for name in ('pooled', 'ensemble'):
        add_margins(result[name], baseline, weights)
    oa, ob = np.broadcast_to(ma, a.shape).copy(), np.broadcast_to(mb, b.shape).copy()
    oa[:, :101], ob[:, :101] = a[:, :101], b[:, :101]
    result['first_leg_oracle'] = block([moments(oa, ob, a, b, pattern)], truth, weights)
    result['first_leg_oracle']['status'] = 'INFERRED attribution control using test truth; NOT a deployable baseline'
    for record in [*result['per_seed'].values(), result['pooled'], result['ensemble']]:
        for name, window in record['windows'].items():
            oracle_s = result['first_leg_oracle']['windows'][name]['contrast']['S']['per_unit']
            window['margin_S_vs_first_leg_oracle'] = summary(window['contrast']['S']['per_unit'] - oracle_s, weights)
    result['truth_scales'] = {'centered_rms_per_unit_frame': np.sqrt(truth['b2']),
                              'raw_contrast_rms_per_unit_frame': np.sqrt(truth['delta2']),
                              'training_units': len(ta), 'unit_indices': np.arange(n)}
    result['scientific_verdict'] = 'not assigned; apply adopted reading rule without changing thresholds'
    return result


def selftest(mutation=None):
    import test_path_vs_exposure as tests

    tests.MUTATION = mutation is not None
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    a, b, ta, tb, bad_a, bad_b, good_a, good_b = tests.fixture()
    demonstrations = {}
    for name, pa, pb in (('recognition_without_composition', bad_a, bad_b),
                         ('good_composition', good_a, good_b)):
        record = tests.run_pair(a, b, ta, tb, pa, pb)
        full = record['per_seed']['0']['windows']['full']
        demonstrations[name] = {
            'raw_sign_agreement': full['recognition']['raw']['mean'],
            'centered_S': full['contrast']['S']['mean'],
            'skill': full['contrast']['skill_mean'],
            'recognition_only_S': record['recognition_only']['windows']['full']['contrast']['S']['mean'],
            'ab_relative_l2': full['arms']['ab']['relative_l2']['mean'],
            'ba_relative_l2': full['arms']['ba']['relative_l2']['mean']}
    print(json.dumps(json_ready({'status': 'MEASURED synthetic only', 'tests_run': outcome.testsRun,
                                 'passed': outcome.wasSuccessful(), 'mutation': mutation,
                                 'network_used': False, 'real_inputs_opened': False,
                                 'examples': demonstrations}), indent=2, allow_nan=False))
    return 0 if outcome.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--mutation', choices=['swap-arms'])
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--confirm-frozen-inputs', action='store_true')
    args = parser.parse_args(argv)
    if args.selftest:
        if args.predictions or args.truth or args.output or args.confirm_frozen_inputs:
            parser.error('--selftest cannot take real-input/output or attestation arguments')
        return selftest(args.mutation)
    if args.mutation:
        parser.error('--mutation requires --selftest')
    if not (args.predictions and args.truth and args.output and args.confirm_frozen_inputs):
        parser.error('provide all input/output paths and --confirm-frozen-inputs only after closure and rule adoption')
    require(args.seeds == [0, 1, 2, 3, 4], 'CLI requires all five Route E seeds in frozen order')
    require(args.output.resolve().is_relative_to(HERE), 'output must be under scripts/route_e')
    require(not args.output.exists() and args.output.parent.is_dir(), 'output must be new in an existing directory')
    a, b, center, truth_info = load_truth(args.truth)
    with np.load(args.truth, allow_pickle=False) as archive:
        ta, tb = numeric(archive['e_train_ab'], 'e_train_ab'), numeric(archive['e_train_ba'], 'e_train_ba')
    require(a.shape == (48, 201, 256) and ta.shape == (160, 201, 256), 'CLI requires Route E cohort dimensions')
    require(np.array_equal(center, (ta - tb).mean(axis=0)), 'truth changed between loads')
    require(receipt(args.truth) == {key: truth_info[key] for key in ('path', 'sha256')}, 'truth receipt changed')
    pa, pb, prediction_info = load_predictions(args.predictions, args.seeds)
    result = analyze(pa, pb, a, b, ta, tb, seeds=args.seeds)
    result['provenance'] = {'status': 'MEASURED supplied frozen captures', 'truth': truth_info,
                            'predictions': prediction_info, 'analyzer': receipt(Path(__file__)),
                            'loader': receipt(HERE / 's_of_t.py'),
                            'network_used': False, 'weights_loaded': False,
                            'frozen_inputs_attested_by_operator': True,
                            'join': 'array order and exact initial-state equality; capture has no embedded unit IDs or times',
                            'timing_source': 'scripts/code/generate.py:44-49,102-106',
                            'numpy': np.__version__, 'python': sys.version.split()[0]}
    with args.output.open('x') as stream:
        json.dump(json_ready(result), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Wrote {args.output}. No scientific verdict assigned.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
