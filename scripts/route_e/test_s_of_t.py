import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import s_of_t as sot


def synthetic(exponent=0.0, units=48, seeds=5):
    frames = 201
    r = np.r_[0.0, np.geomspace(0.005, 0.25, frames - 1)]
    pattern = np.broadcast_to(0.03 * (np.arange(frames) > 0)[:, None], (frames, 4))
    signs = np.where(np.arange(units) % 2, -1.0, 1.0)
    delta = pattern[None] + signs[:, None, None] * r[None, :, None]
    truth_ab = np.ones((units, frames, 4))
    truth_ba = truth_ab - delta
    epsilon = np.r_[0.0, 0.02 * (r[1:] / 0.02) ** exponent]
    offsets = np.linspace(0.8, 1.2, seeds) if seeds > 1 else np.ones(1)
    pred_ab = truth_ab[None] + offsets[:, None, None, None] * epsilon[None, None, :, None]
    pred_ba = np.broadcast_to(truth_ba, pred_ab.shape).copy()
    return pred_ab, pred_ba, truth_ab, truth_ba, pattern


class SOfTTests(unittest.TestCase):
    def analyze(self, exponent=0.0):
        return sot.analyze(*synthetic(exponent), bootstrap=160, rng_seed=19)

    def test_constant_error_inverse_law_and_crossings(self):
        result = self.analyze()
        for block in [*result['per_seed'].values(), result['ensemble'], result['seed_error_pool']]:
            fit = block['fits']['pooled']
            self.assertEqual(f"{fit['slope']:.3f}", '-1.000')
            np.testing.assert_allclose(fit['slope_ci95'], [-1, -1], atol=1e-12)
            curve = block['curves']
            r, eps, score = (np.asarray(curve[k]) for k in ('r_pool', 'epsilon_pool', 'S_pool'))
            np.testing.assert_allclose(score[1:], eps[1:] / r[1:], atol=1e-12)
            self.assertTrue(np.isnan(score[0]))
            self.assertEqual(fit['n_frames'], 200)
            self.assertEqual(len(block['crossings']['pooled']), 1)
            crossing = block['crossings']['pooled'][0]
            self.assertAlmostEqual(crossing['r'], crossing['epsilon'], places=12)
            self.assertAlmostEqual(fit['r_at_S1'], eps[1], places=12)
            self.assertAlmostEqual(crossing['r'], eps[1], places=12)

    def test_non_inverse_law_is_detected_without_breaking_identity(self):
        result = self.analyze(0.5)
        fit = result['ensemble']['fits']['pooled']
        self.assertAlmostEqual(fit['slope'], -0.5, places=12)
        self.assertGreater(fit['slope_ci95'][0], -1)
        self.assertGreater(abs(fit['slope'] + 1), 0.4)
        self.assertAlmostEqual(fit['epsilon_slope'], 0.5, places=12)
        self.assertLess(fit['identity_log_max_abs'], 1e-12)

    def test_same_truth_center_does_not_remove_prediction_bias(self):
        pa, pb, a, b, pattern = synthetic()
        metrics = sot.unit_metrics(pa[2], pb[2], a, b, pattern)
        np.testing.assert_allclose(metrics['q2'][:, 1:], 0.02**2, atol=1e-15)
        np.testing.assert_allclose(metrics['b2'][:, 1:], (a - b - pattern)[..., 0][:, 1:]**2)
        self.assertGreater(metrics['q2'][:, 1:].mean(), 0)

    def test_no_structure_baseline_and_shared_scale(self):
        pa, pb, a, b, pattern = synthetic()
        symmetric = 0.5 * (a + b)
        baseline = sot.unit_metrics(symmetric + pattern / 2, symmetric - pattern / 2,
                                    a, b, pattern)
        c = sot.curves(baseline['q2'], baseline['b2'], baseline['y2'])
        np.testing.assert_allclose(c['S_mean'][1:], 1, atol=1e-12)
        np.testing.assert_allclose(c['margin_vs_common_pattern'][1:], 0, atol=1e-12)
        metrics = sot.unit_metrics(pa[0], pb[0], a, b, pattern)
        scaled = sot.unit_metrics(7 * pa[0], 7 * pb[0], 7 * a, 7 * b, 7 * pattern)
        original = sot.curves(metrics['q2'], metrics['b2'], metrics['y2'])
        transformed = sot.curves(scaled['q2'], scaled['b2'], scaled['y2'])
        for key in ('r_pool', 'epsilon_pool', 'S_pool', 'S_mean'):
            np.testing.assert_allclose(original[key], transformed[key], rtol=1e-12)

    def test_mean_ratios_and_energy_pool_are_distinct(self):
        q2 = np.array([[1.0, 1.0], [1.0, 1.0]])
        b2 = np.array([[1.0, 1.0], [9.0, 9.0]])
        y2 = np.ones_like(b2)
        c = sot.curves(q2, b2, y2)
        np.testing.assert_allclose(c['S_mean'], 2 / 3)
        np.testing.assert_allclose(c['S_pool'], np.sqrt(0.2))
        np.testing.assert_allclose(c['margin_vs_common_pattern'], -1 / 3)

    def test_ensemble_is_not_average_seed_error(self):
        pa, pb, a, b, pattern = synthetic(seeds=2)
        pa[0] = a + 0.01
        pa[1] = a - 0.01
        pa[:, :, 0] = a[:, 0]
        result = sot.analyze(pa, pb, a, b, pattern, bootstrap=100)
        np.testing.assert_allclose(result['ensemble']['curves']['S_pool'][1:], 0, atol=1e-12)
        self.assertGreater(result['seed_error_pool']['curves']['S_pool'][1], 0)
        self.assertEqual(result['ensemble']['fits']['pooled']['status'], 'void')

    def test_unit_bootstrap_preserves_whole_curves_and_seed_pairing(self):
        pa, pb, a, b, pattern = synthetic()
        r = np.r_[0.0, np.geomspace(0.005, 0.25, 200)]
        exponents = np.linspace(-0.3, 0.3, len(a))
        error = 0.02 * (r[None, 1:] / 0.02) ** exponents[:, None]
        pa[:, :, 1:] = a[None, :, 1:] + error[None, :, :, None]
        first = sot.analyze(pa, pb, a, b, pattern, bootstrap=180, rng_seed=22)
        second = sot.analyze(pa, pb, a, b, pattern, bootstrap=180, rng_seed=22)
        f = first['ensemble']['fits']['pooled']
        self.assertGreater(f['slope_ci95'][1] - f['slope_ci95'][0], 0.01)
        self.assertEqual(sot.json_ready(f), sot.json_ready(second['ensemble']['fits']['pooled']))
        self.assertEqual(first['uncertainty']['independent_units'], 48)
        np.testing.assert_array_equal(first['per_seed']['0']['fits']['pooled']['slope_ci95'],
                                      first['per_seed']['1']['fits']['pooled']['slope_ci95'])

    def test_undefined_denominators_and_no_extrapolated_crossings(self):
        q = np.ones((2, 201))
        b = q.copy()
        b[0, 5] = 0
        c = sot.curves(q, b, q)
        self.assertTrue(np.isnan(c['S_mean'][5]))
        self.assertTrue(np.isfinite(c['S_pool'][5]))
        t = np.arange(4, dtype=float)
        self.assertEqual(sot.crossings(t, np.ones(4) * 2, np.ones(4), np.ones(4)), [])
        crossed = sot.crossings(t, np.array([2., 0.5, 2., 0.5]), np.ones(4), np.ones(4))
        self.assertEqual(len(crossed), 3)
        plateaus = sot.crossings(t, np.ones(4), np.ones(4), np.ones(4))
        self.assertEqual(len(plateaus), 1)
        self.assertEqual(plateaus[0]['kind'], 'plateau')

    def test_trajectory_norms_precede_unit_ratios(self):
        pa, pb, a, b, pattern = synthetic()
        result = sot.analyze(pa, pb, a, b, pattern, bootstrap=100)
        d = a - b - pattern
        q = pa[0] - pb[0] - (a - b)
        expected = np.mean(np.sqrt(np.sum(q*q, axis=(1, 2)) / np.sum(d*d, axis=(1, 2))))
        self.assertAlmostEqual(result['per_seed']['0']['observables']['trajectory']['S_mean'], expected)

    def test_invalid_inputs_fail(self):
        data = list(synthetic())
        data[0][0, 0, 4, 1] = np.nan
        with self.assertRaises(ValueError):
            sot.analyze(*data, bootstrap=100)
        data = list(synthetic())
        data[0] = data[0][:, :, 1:]
        with self.assertRaises(ValueError):
            sot.analyze(*data, bootstrap=100)
        data = list(synthetic())
        data[1][0, 0, 0, 0] += 1
        with self.assertRaises(ValueError):
            sot.analyze(*data, bootstrap=100)

    def test_loaders_and_truth_training_mean(self):
        pa, pb, a, b, pattern = synthetic()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='sot_test_') as tmp:
            root = Path(tmp)
            for seed in range(5):
                np.savez(root / f'composed_seed{seed}.npz', pred_ab=pa[seed], pred_ba=pb[seed])
            loaded_a, loaded_b, receipt = sot.load_predictions(root, list(range(5)))
            np.testing.assert_array_equal(loaded_a, pa)
            np.testing.assert_array_equal(loaded_b, pb)
            self.assertEqual(len(receipt), 5)
            with self.assertRaises(FileNotFoundError):
                sot.load_predictions(root, list(range(6)))
            stacked = root / 'stacked.npz'
            np.savez(stacked, pred_ab=pa, pred_ba=pb, model_seeds=np.arange(5))
            np.testing.assert_array_equal(sot.load_predictions(stacked, list(range(5)))[0], pa)
            with self.assertRaises(ValueError):
                sot.load_predictions(stacked, [4, 3, 2, 1, 0])
            npy_root = root / 'npy'
            npy_root.mkdir()
            for seed in range(5):
                np.save(npy_root / f'pred_ab_seed{seed}.npy', pa[seed])
                np.save(npy_root / f'pred_ba_seed{seed}.npy', pb[seed])
            np.testing.assert_array_equal(sot.load_predictions(npy_root, list(range(5)))[0], pa)
            truth = root / 'truth.npz'
            training_pattern = 3 * pattern
            np.savez(truth, e_test_ab=a, e_test_ba=b,
                     e_train_ab=np.ones((160, 201, 4)),
                     e_train_ba=np.ones((160, 201, 4)) - training_pattern)
            truth_a, truth_b, center, info = sot.load_truth(truth)
            np.testing.assert_array_equal(truth_a, a)
            np.testing.assert_array_equal(truth_b, b)
            np.testing.assert_allclose(center, training_pattern, atol=1e-15)
            self.assertGreater(np.max(np.abs(center - (a-b).mean(axis=0))), 0.05)
            self.assertEqual(info['training_units'], 160)

    def test_power_is_conditional_not_201_independent_samples(self):
        p = sot.power_design(48)
        self.assertIsNone(p['design_only_mde'])
        self.assertAlmostEqual(p['mde_per_unit_influence_sd'], 2.801585218 / np.sqrt(48), places=8)

    def test_cli_roundtrip_guard_and_headless_figure(self):
        here = Path(__file__).parent
        with tempfile.TemporaryDirectory(dir=here, prefix='sot_cli_test_') as tmp:
            root = Path(tmp)
            pa, pb, a, b, pattern = synthetic(seeds=1)
            truth = root / 'truth.npz'
            prediction = root / 'composed_seed0.npz'
            expand = lambda value: np.tile(value.astype(np.float32), (1,) * (value.ndim - 1) + (64,))
            np.savez_compressed(truth, e_test_ab=expand(a), e_test_ba=expand(b),
                                e_train_ab=np.ones((160, 201, 256), dtype=np.float32),
                                e_train_ba=expand(np.broadcast_to(1 - pattern, (160, 201, 4))))
            np.savez_compressed(prediction, pred_ab=expand(pa[0]), pred_ba=expand(pb[0]))
            command = [sys.executable, '-B', str(here / 's_of_t.py'), '--predictions', str(prediction),
                       '--truth', str(truth), '--seeds', '0', '--bootstrap', '100']
            env = {**os.environ, 'MPLCONFIGDIR': str(root / 'mpl'), 'PYTHONDONTWRITEBYTECODE': '1'}
            denied = subprocess.run(command + ['--output', str(root / 'denied.json')],
                                    capture_output=True, text=True, env=env)
            self.assertNotEqual(denied.returncode, 0)
            self.assertFalse((root / 'denied.json').exists())
            outputs = [root / f'analysis{i}.json' for i in range(2)]
            for output in outputs:
                done = subprocess.run(command + ['--output', str(output), '--confirm-frozen-inputs'],
                                      capture_output=True, text=True, env=env)
                self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            payload = json.loads(outputs[0].read_text())
            self.assertIsNone(payload['ensemble']['curves']['S_pool'][0])
            self.assertAlmostEqual(payload['ensemble']['fits']['pooled']['slope'], -1, places=5)
            self.assertEqual(payload['provenance']['truth']['training_units'], 160)
            payload['synthetic_only'] = True
            synthetic_json = root / 'synthetic.json'
            synthetic_json.write_text(json.dumps(payload, allow_nan=False))
            hashes = []
            for index in range(2):
                image = root / f'figure{index}.png'
                plotted = subprocess.run([sys.executable, '-B', str(here / 'plot_s_of_t.py'),
                                          str(synthetic_json), '--output', str(image)],
                                         capture_output=True, text=True, env=env)
                self.assertEqual(plotted.returncode, 0, plotted.stderr)
                self.assertTrue(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                hashes.append(hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(hashes[0], hashes[1])
            overwritten = subprocess.run(command + ['--output', str(outputs[0]), '--confirm-frozen-inputs'],
                                         capture_output=True, text=True, env=env)
            self.assertNotEqual(overwritten.returncode, 0)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())

    def test_assertions_catch_a_broken_score(self):
        original = sot.curves

        def wrong(*args, **kwargs):
            result = original(*args, **kwargs)
            result['S_pool'] = np.ones_like(result['S_pool'])
            return result

        with patch.object(sot, 'curves', wrong):
            with self.assertRaises(AssertionError):
                self.test_constant_error_inverse_law_and_crossings()


if __name__ == '__main__':
    unittest.main()
