from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
TIMES = np.arange(201, dtype=np.float64) * 0.01


def require(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def numeric(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in 'fiu', f'{name}: real numeric arrays required')
    require(np.isfinite(array).all(), f'{name}: nonfinite input')
    return array.astype(np.float64)


def load_predictions(path, seeds):
    path = Path(path)
    require(len(seeds) > 0 and len(set(seeds)) == len(seeds), 'unique seeds required')
    files, first, second = [], [], []
    if path.is_file():
        require(path.suffix == '.npz', 'single-file predictions must be NPZ')
        with np.load(path, allow_pickle=False) as archive:
            a = numeric(archive['pred_ab'], 'pred_ab')
            b = numeric(archive['pred_ba'], 'pred_ba')
            if a.ndim == 3:
                require(len(seeds) == 1, 'one unstacked NPZ needs exactly one seed')
                match = re.fullmatch(r'composed_seed(\d+)\.npz', path.name)
                require(not match or int(match[1]) == seeds[0], 'filename seed mismatch')
                a, b = a[None], b[None]
            else:
                require('model_seeds' in archive, 'stacked NPZ needs model_seeds')
                require(np.array_equal(archive['model_seeds'], seeds), 'seed order mismatch')
            if 'model_seeds' in archive:
                require(np.array_equal(archive['model_seeds'], seeds), 'seed identity mismatch')
        files.append(receipt(path))
    else:
        require(path.is_dir(), 'prediction path must exist')
        for seed in seeds:
            packed = path / f'composed_seed{seed}.npz'
            a_file, b_file = (path / f'pred_{arm}_seed{seed}.npy' for arm in ('ab', 'ba'))
            require(not (packed.exists() and (a_file.exists() or b_file.exists())),
                    f'ambiguous NPZ/NPY representations for seed {seed}')
            if packed.exists():
                with np.load(packed, allow_pickle=False) as archive:
                    first.append(numeric(archive['pred_ab'], 'pred_ab'))
                    second.append(numeric(archive['pred_ba'], 'pred_ba'))
                files.append(receipt(packed))
            else:
                for target, filename in ((first, a_file), (second, b_file)):
                    with filename.open('rb') as stream:
                        target.append(numeric(np.load(stream, allow_pickle=False), str(filename)))
                    files.append(receipt(filename))
        a, b = np.stack(first), np.stack(second)
    require(a.ndim == 4 and a.shape == b.shape and a.shape[0] == len(seeds),
            'predictions must have equal [seed, unit, frame, space] shapes')
    return a, b, files


def load_truth(path):
    with np.load(path, allow_pickle=False) as archive:
        a, b, train_a, train_b = (
            numeric(archive[k], k) for k in
            ('e_test_ab', 'e_test_ba', 'e_train_ab', 'e_train_ba')
        )
    require(a.ndim == 3 and a.shape == b.shape, 'truth must be [unit, frame, space]')
    require(train_a.ndim == 3 and train_a.shape == train_b.shape
            and train_a.shape[1:] == a.shape[1:] and len(train_a) > 0,
            'evaluator-training arms must match the truth frame/space axes')
    require(np.array_equal(train_a[:, 0], train_b[:, 0]), 'training initial arms differ')
    info = {**receipt(path), 'training_units': len(train_a), 'test_units': len(a),
            'centering': 'float64 evaluator-TRAIN AB-minus-BA mean, fixed in bootstrap'}
    return a, b, (train_a - train_b).mean(axis=0), info


def ratio(numerator, denominator):
    shape = np.broadcast_shapes(np.shape(numerator), np.shape(denominator))
    return np.divide(numerator, denominator, out=np.full(shape, np.nan),
                     where=np.asarray(denominator) > 0)


def unit_metrics(pred_ab, pred_ba, truth_ab, truth_ba, pattern):
    delta = truth_ab - truth_ba
    centered_true = delta - pattern
    centered_pred = pred_ab - pred_ba - pattern
    residual = centered_pred - centered_true
    error_ab, error_ba = pred_ab - truth_ab, pred_ba - truth_ba
    return {'q2': np.mean(residual**2, axis=-1),
            'b2': np.mean(centered_true**2, axis=-1),
            'y2': np.mean(truth_ab**2, axis=-1),
            'error_ab2': np.mean(error_ab**2, axis=-1),
            'error_ba2': np.mean(error_ba**2, axis=-1),
            'error_ab_ba': np.mean(error_ab * error_ba, axis=-1)}


def unit_mean(values, weights=None):
    if weights is None:
        return values.mean(axis=0)
    finite = np.isfinite(values)
    result = weights @ np.where(finite, values, 0.0)
    return np.where(weights @ (~finite).astype(float) > 0, np.nan, result)


def curves(q2, b2, y2, weights=None):
    q, b, y = (np.sqrt(unit_mean(v, weights)) for v in (q2, b2, y2))
    s_mean = unit_mean(ratio(np.sqrt(q2), np.sqrt(b2)), weights)
    return {'q_pool': q, 'b_pool': b, 'Y_pool': y,
            'r_pool': ratio(b, y), 'epsilon_pool': ratio(q, y),
            'S_pool': ratio(q, b), 'S_mean': s_mean,
            'margin_vs_common_pattern': s_mean - 1.0,
            'valid_mean_units': np.sum(b2 > 0, axis=0)}


def regression(r, score):
    r, score = np.atleast_2d(r), np.atleast_2d(score)
    r, score = np.broadcast_arrays(r, score)
    good = (r > 0) & (score > 0) & np.isfinite(r) & np.isfinite(score)
    good[:, 0] = False
    n = good.sum(axis=1)
    x, y = np.zeros_like(r), np.zeros_like(score)
    np.log(r, out=x, where=good)
    np.log(score, out=y, where=good)
    mx, my = ratio(x.sum(axis=1), n), ratio(y.sum(axis=1), n)
    dx, dy = np.where(good, x - mx[:, None], 0), np.where(good, y - my[:, None], 0)
    sxx = np.sum(dx * dx, axis=1)
    slope = ratio(np.sum(dx * dy, axis=1), sxx)
    intercept = my - slope * mx
    valid = (n == r.shape[1] - 1) & (sxx > 1e-12)
    return np.where(valid, slope, np.nan), np.where(valid, intercept, np.nan), n


def power_design(units):
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
    return {'status': 'INFERRED conditional normal-approximation sensitivity, not measured power',
            'independent_units': units, 'nominal_frames': 201, 'positive_fit_frames': 200,
            'alpha_two_sided': 0.05, 'target_power': 0.8, 'normal_multiplier': z,
            'design_only_mde': None, 'mde_per_unit_influence_sd': z / math.sqrt(units),
            'sensitivity': [{'unit_influence_sd': sd, 'mde': z * sd / math.sqrt(units)}
                            for sd in (0.1, 0.25, 0.5, 1.0)],
            'reason': 'Frame count and unit count do not identify residual covariance or slope variance.'}


def fitted(r, score, epsilon, resampled, score_key):
    slopes, intercepts, counts = regression(r, score)
    beta, alpha = float(slopes[0]), float(intercepts[0])
    boot_b, boot_a, _ = regression(resampled['r_pool'], resampled[score_key])
    valid = np.isfinite(boot_b) & np.isfinite(boot_a)
    usable = bool(np.isfinite(beta) and np.isfinite(alpha) and valid.all())
    result = {'status': 'ok' if usable else 'void', 'slope': beta, 'intercept': alpha,
              'n_frames': int(counts[0]), 'log_base': 'natural',
              'bootstrap_valid': int(valid.sum()), 'bootstrap_total': len(valid),
              'slope_ci95': None, 'slope_ci90': None, 'intercept_ci95': None,
              'slope_bootstrap_se': None, 'mde80_normal_approx': None,
              'r_at_S1': None, 'crossing_in_observed_r_range': False}
    if usable:
        result.update(slope_ci95=np.quantile(boot_b, [0.025, 0.975]),
                      slope_ci90=np.quantile(boot_b, [0.05, 0.95]),
                      intercept_ci95=np.quantile(boot_a, [0.025, 0.975]),
                      slope_bootstrap_se=float(np.std(boot_b, ddof=1)))
        result['mde80_normal_approx'] = (
            power_design(1)['normal_multiplier'] * result['slope_bootstrap_se'])
    mask = (r > 0) & (score > 0) & (epsilon > 0)
    mask[0] = False
    if np.isfinite(beta) and abs(beta) > 1e-12:
        log_cross = -alpha / beta
        if -700 < log_cross < 700:
            result['r_at_S1'] = math.exp(log_cross)
            result['crossing_in_observed_r_range'] = bool(
                r[mask].min() <= result['r_at_S1'] <= r[mask].max())
    if mask.any():
        le, lr, ls = np.log(epsilon[mask]), np.log(r[mask]), np.log(score[mask])
        result['descriptive_constant_epsilon'] = float(np.exp(le.mean()))
        result['epsilon_max_over_min'] = float(np.exp(le.max() - le.min()))
        if score_key == 'S_pool':
            eps_beta, _, _ = regression(r, epsilon)
            result['epsilon_slope'] = float(eps_beta[0])
            result['identity_log_max_abs'] = float(np.max(np.abs(ls - le + lr)))
            result['slope_decomposition_error'] = beta + 1 - float(eps_beta[0])
        result['log_fit_rmse'] = float(np.sqrt(np.mean((ls - alpha - beta * lr)**2)))
    return result


def crossings(times, score, r, epsilon):
    times, score, r, epsilon = map(np.asarray, (times, score, r, epsilon))
    valid = np.isfinite(score) & (score > 0) & (r > 0) & (epsilon > 0)
    z = np.full(score.shape, np.nan)
    np.log(score, out=z, where=valid)
    result = []
    i = 0
    while i < len(times):
        if valid[i] and abs(z[i]) <= 1e-12:
            end = i
            while end + 1 < len(times) and valid[end + 1] and abs(z[end + 1]) <= 1e-12:
                end += 1
            result.append({'kind': 'plateau' if end > i else 'exact_frame',
                           'frame_bracket': [i, end], 'time_interval': [times[i], times[end]],
                           't': times[i], 'r': r[i], 'epsilon': epsilon[i],
                           'r_minus_epsilon': r[i] - epsilon[i]})
            i = end + 1
            continue
        if i + 1 < len(times) and valid[i:i+2].all() and z[i] * z[i+1] < 0:
            if abs(z[i+1]) > 1e-12:
                fraction = -z[i] / (z[i+1] - z[i])
                rr, ee = (float(np.exp(np.log(v[i]) + fraction * np.log(v[i+1] / v[i])))
                          for v in (r, epsilon))
                result.append({'kind': 'log_linear_interpolation', 'frame_bracket': [i, i+1],
                               't': times[i] + fraction * (times[i+1] - times[i]),
                               'r': rr, 'epsilon': ee, 'r_minus_epsilon': rr - ee,
                               'direction': 'below_to_above' if z[i] < 0 else 'above_to_below'})
        i += 1
    return result


def block(q2, b2, y2, weights):
    curve = curves(q2, b2, y2)
    sampled = curves(q2, b2, y2, weights)
    result = {'curves': curve,
              'units': {'q': np.sqrt(q2), 'epsilon': ratio(np.sqrt(q2), np.sqrt(y2)),
                        'S': ratio(np.sqrt(q2), np.sqrt(b2))},
              'fits': {}, 'crossings': {}, 'observables': {}}
    for label, key in (('pooled', 'S_pool'), ('mean_score', 'S_mean')):
        result['fits'][label] = fitted(curve['r_pool'], curve[key], curve['epsilon_pool'], sampled, key)
        result['crossings'][label] = crossings(TIMES, curve[key], curve['r_pool'], curve['epsilon_pool'])
    result['curves']['margin_ci95_pointwise'] = np.quantile(
        sampled['margin_vs_common_pattern'], [0.025, 0.975], axis=0)
    for name, arrays in (
        ('trajectory', [v.mean(axis=1, keepdims=True) for v in (q2, b2, y2)]),
        ('endpoint', [v[:, -1:] for v in (q2, b2, y2)]),
    ):
        summary = curves(*arrays)
        summary.pop('valid_mean_units')
        result['observables'][name] = {k: float(v[0]) for k, v in summary.items()}
        boot_margin = curves(*arrays, weights)['margin_vs_common_pattern'][:, 0]
        result['observables'][name]['margin_ci95'] = np.quantile(boot_margin, [0.025, 0.975])
    return result


def analyze(pred_ab, pred_ba, truth_ab, truth_ba, pattern, bootstrap=4000,
            rng_seed=20260905, seeds=None):
    pred_ab, pred_ba, truth_ab, truth_ba, pattern = (
        numeric(a, name) for a, name in zip(
            (pred_ab, pred_ba, truth_ab, truth_ba, pattern),
            ('pred_ab', 'pred_ba', 'truth_ab', 'truth_ba', 'pattern')))
    require(pred_ab.ndim == 4 and pred_ab.shape == pred_ba.shape,
            'prediction shape must be [seed, unit, frame, space]')
    require(truth_ab.ndim == 3 and truth_ab.shape == truth_ba.shape == pred_ab.shape[1:],
            'truth/prediction axes differ')
    k, units, frames, points = pred_ab.shape
    require(units >= 2 and k >= 1 and points >= 1 and frames == 201, 'need 201 frames and >=2 units')
    require(pattern.shape == (frames, points), 'training mean shape differs')
    require(np.array_equal(truth_ab[:, 0], truth_ba[:, 0]), 'truth initial arms differ')
    require(np.array_equal(pred_ab[:, :, 0], np.broadcast_to(truth_ab[:, 0], (k, units, points)))
            and np.array_equal(pred_ba[:, :, 0], pred_ab[:, :, 0]), 'prediction initial states differ')
    require(np.all(pattern[0] == 0), 'training contrast at initial frame is nonzero')
    require(bootstrap >= 100, 'need at least 100 bootstrap replicates')
    seeds = list(range(k)) if seeds is None else list(seeds)
    require(len(seeds) == k and len(set(seeds)) == k, 'seed IDs mismatch')
    weights = np.random.default_rng(rng_seed).multinomial(units, np.ones(units) / units, bootstrap) / units
    b2 = np.mean((truth_ab - truth_ba - pattern)**2, axis=-1)
    y2 = np.mean(truth_ab**2, axis=-1)
    require(np.isfinite(b2).all() and np.isfinite(y2).all(), 'truth norm overflow')
    result = {'schema': 'pde-s-of-t-v1', 'times': TIMES, 'frame_indices': np.arange(frames),
              'model_seeds': seeds, 'shape': list(pred_ab.shape),
              'centering': 'same evaluator-training truth mean subtracted from both contrasts',
              'units': {'b': np.sqrt(b2), 'Y': np.sqrt(y2), 'r': ratio(np.sqrt(b2), np.sqrt(y2)),
                        'initial_sha256': [hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest()
                                           for u in truth_ab[:, 0]]},
              'uncertainty': {'independent_units': units, 'bootstrap_replicates': bootstrap,
                              'rng_seed': rng_seed, 'resample': 'paired whole units across all frames and seeds',
                              'conditional_on': 'fixed training mean, fixed fitted models, fixed demonstration policy',
                              'seed_axis_is_not_independent_units': True},
              'power_design': power_design(units), 'per_seed': {}}
    q2s = []
    for i, seed in enumerate(seeds):
        moments = unit_metrics(pred_ab[i], pred_ba[i], truth_ab, truth_ba, pattern)
        require(all(np.isfinite(v).all() for v in moments.values()), 'prediction norm overflow')
        item = block(moments['q2'], b2, y2, weights)
        item['arm_error_moments'] = {name: moments[name].mean(axis=0)
                                     for name in ('error_ab2', 'error_ba2', 'error_ab_ba')}
        result['per_seed'][str(seed)] = item
        q2s.append(moments['q2'])
    pooled_q2 = np.mean(q2s, axis=0)
    result['seed_error_pool'] = block(pooled_q2, b2, y2, weights)
    result['seed_error_pool']['estimand'] = 'equal-seed residual energy, then mean-unit or pooled norms; not an ensemble'
    ensemble = unit_metrics(pred_ab.mean(axis=0), pred_ba.mean(axis=0), truth_ab, truth_ba, pattern)
    result['ensemble'] = block(ensemble['q2'], b2, y2, weights)
    result['ensemble']['estimand'] = 'predictions averaged across seeds before scoring, as in train_eval_v2'
    result['mean_seed_score'] = np.mean([v['curves']['S_mean'] for v in result['per_seed'].values()], axis=0)
    return result


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def selftest():
    import test_s_of_t

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_s_of_t)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'synthetic_only': True, 'network_used': False,
                      'tests_run': outcome.testsRun, 'passed': outcome.wasSuccessful()}))
    return 0 if outcome.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description='Offline centered S(t); never imports training or loads weights.')
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--rng-seed', type=int, default=20260905)
    parser.add_argument('--confirm-frozen-inputs', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        if args.predictions or args.truth or args.output:
            parser.error('--selftest cannot be combined with real-input or output arguments')
        return selftest()
    if not (args.predictions and args.truth and args.output and args.confirm_frozen_inputs):
        parser.error('provide --predictions, --truth, --output and --confirm-frozen-inputs after closure/freeze')
    require(not args.output.exists(), 'output exists; refusing overwrite')
    require(args.output.parent.is_dir(), 'output parent must already exist')
    a, b, center, truth_info = load_truth(args.truth)
    pa, pb, prediction_info = load_predictions(args.predictions, args.seeds)
    require(a.shape == (48, 201, 256) and truth_info['training_units'] == 160,
            'CLI freezes Route E at 48 test units, 160 training units, 201 frames, 256 points')
    result = analyze(pa, pb, a, b, center, args.bootstrap, args.rng_seed, args.seeds)
    result['provenance'] = {'truth': truth_info, 'predictions': prediction_info,
                            'analyzer': receipt(Path(__file__)), 'network_used': False,
                            'frozen_inputs_attested_by_operator': True,
                            'numpy': np.__version__, 'python': sys.version.split()[0],
                            'frame_timing': 'generate.py: dt=0.01, 100 steps per leg; capture has no embedded timing',
                            'join': 'exact initial-state equality and array order; no saved per-frame identity manifest'}
    with args.output.open('x') as stream:
        json.dump(json_ready(result), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Wrote {args.output}; plot with plot_s_of_t.py. No scientific verdict is assigned automatically.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
