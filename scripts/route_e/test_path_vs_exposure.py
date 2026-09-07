"""Synthetic-only checks. No corpus, model, or run output is opened."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import path_vs_exposure as p9


def fixture():
    t = np.arange(201, dtype=float)[None, :, None] / 200
    c = np.linspace(-0.3, 0.3, 8)[:, None, None]
    x = np.array([-1.0, 1.0] * 4)[None, None, :]
    common = 2 + c * x * (1 + 8 * t)
    delta = 0.8 * t * (1 + 0.5 * c * x)
    a, b = common + delta / 2, common - delta / 2
    ta, tb = a.copy(), b.copy()
    ma, mb = ta.mean(axis=0), tb.mean(axis=0)
    bad_a = np.broadcast_to(ma, a.shape).copy() + 3 * t
    bad_b = np.broadcast_to(mb, b.shape).copy() + 3 * t
    bad_a[:, 0], bad_b[:, 0] = a[:, 0], b[:, 0]
    d = delta - delta.mean(axis=0)
    good_a, good_b = a + 0.01 * d + 0.001 * t, b - 0.01 * d + 0.001 * t
    return a, b, ta, tb, bad_a, bad_b, good_a, good_b


def run_pair(a, b, ta, tb, pa, pb):
    return p9.analyze(pa[None], pb[None], a, b, ta, tb, seeds=[0], bootstrap=100)


def assert_good(case, record):
    full = record['per_seed']['0']['windows']['full']
    case.assertLess(full['contrast']['S']['mean'], 0.021)
    for arm in ('ab', 'ba'):
        case.assertLess(full['arms'][arm]['relative_l2']['mean'], 0.01)
    case.assertGreater(full['recognition']['raw']['mean'], 0.99)


class PathExposureTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.ta, self.tb, self.bad_a, self.bad_b, self.good_a, self.good_b = fixture()

    def analyze(self, pa, pb, **kwargs):
        return p9.analyze(pa, pb, self.a, self.b, self.ta, self.tb,
                          bootstrap=100, **kwargs)

    def test_recognition_can_coexist_with_bad_composition(self):
        result = self.analyze(self.bad_a[None], self.bad_b[None])
        full = result['per_seed']['0']['windows']['full']
        baseline = result['recognition_only']['windows']['full']
        self.assertGreater(full['recognition']['raw']['mean'], 0.99)
        for window in ('full', 'second_leg', 'endpoint'):
            for arm in ('ab', 'ba'):
                self.assertGreater(result['per_seed']['0']['windows'][window]['arms'][arm]
                                   ['relative_l2']['mean'], 0.3)
        self.assertAlmostEqual(full['contrast']['S']['mean'], 1.0)
        self.assertAlmostEqual(full['contrast']['S']['mean'], baseline['contrast']['S']['mean'])

    def test_good_composition_beats_recognition_only(self):
        pa, pb = self.good_a, self.good_b
        if MUTATION:
            pa, pb = pb, pa
        result = self.analyze(pa[None], pb[None])
        assert_good(self, result)
        for name, window in result['per_seed']['0']['windows'].items():
            self.assertGreater(window['contrast']['skill_mean'], 0.97, name)
            self.assertAlmostEqual(result['recognition_only']['windows'][name]['contrast']['S']['mean'], 1.0)
            for arm in ('ab', 'ba'):
                self.assertLess(window['arms'][arm]['relative_l2']['mean'], 0.01)
                self.assertLess(window['margins_vs_recognition_only']['arms'][arm]['ci95'][1], 0)

    def test_arm_swap_mutation_is_detected(self):
        result = self.analyze(self.good_b[None], self.good_a[None])
        with self.assertRaises(AssertionError):
            assert_good(self, result)
        self.assertEqual(result['per_seed']['0']['windows']['full']['recognition']['raw']['mean'], 0)

    def test_center_is_training_only_and_common_contrast_bias_remains(self):
        pa, pb = self.good_a.copy(), self.good_b.copy()
        pa[:, 1:] += 0.2
        result = self.analyze(pa[None], pb[None])
        pattern = (self.ta - self.tb).mean(axis=0)
        d = self.a - self.b - pattern
        expected = np.linalg.norm((pa - pb - self.a + self.b).reshape(8, -1), axis=1)
        expected /= np.linalg.norm(d.reshape(8, -1), axis=1)
        np.testing.assert_allclose(result['per_seed']['0']['windows']['full']['contrast']['S']['per_unit'], expected)
        shifted_ta = self.ta.copy()
        shifted_ta[:, 1:] += 0.1
        changed = p9.analyze(pa[None], pb[None], self.a, self.b, shifted_ta,
                             self.tb, bootstrap=100)
        self.assertNotAlmostEqual(changed['per_seed']['0']['windows']['full']['contrast']['S']['mean'], expected.mean())

    def test_common_mode_errors_do_not_imply_contrast_errors(self):
        t = np.arange(201)[None, :, None] / 200
        result = self.analyze((self.a + 20 * t)[None], (self.b + 20 * t)[None])
        full = result['per_seed']['0']['windows']['full']
        self.assertLess(full['contrast']['S']['mean'], 1e-12)
        self.assertGreater(full['arms']['ab']['relative_l2']['mean'], 1)

    def test_exposures_and_disjoint_energy_accounting(self):
        result = self.analyze(self.bad_a[None], self.bad_b[None])
        exposure = result['exposure']
        self.assertEqual(exposure['equal_exposure_frames'], [0, 200])
        np.testing.assert_allclose(exposure['ab'][150], [1, 0.5])
        np.testing.assert_allclose(exposure['ba'][150], [0.5, 1])
        self.assertEqual(result['windows']['first_leg'], [0, 100])
        self.assertEqual(result['windows']['second_leg'], [100, 200])
        self.assertEqual(result['windows']['post_handoff'], [101, 200])
        energy = result['per_seed']['0']['energy_partition']
        self.assertAlmostEqual(sum(energy['signal_fraction'].values()), 1)
        self.assertAlmostEqual(sum(energy['residual_fraction'].values()), 1)

    def test_pool_is_not_ensemble_and_units_are_not_multiplied(self):
        t = np.arange(201)[None, :, None] / 200
        pa = np.stack([self.a + t, self.a - t])
        pb = np.stack([self.b - t, self.b + t])
        result = self.analyze(pa, pb, seeds=[12, 13])
        pool = result['pooled']['windows']['full']['contrast']['S']
        ensemble = result['ensemble']['windows']['full']['contrast']['S']
        self.assertGreater(pool['mean'], 1)
        self.assertLess(ensemble['mean'], 1e-12)
        self.assertEqual(pool['total_units'], 8)
        self.assertEqual(result['uncertainty']['independent_units'], 8)
        duplicate = self.analyze(np.repeat(pa[:1], 2, axis=0), np.repeat(pb[:1], 2, axis=0))
        np.testing.assert_allclose(duplicate['pooled']['windows']['full']['contrast']['S']['ci95'],
                                   duplicate['per_seed']['0']['windows']['full']['contrast']['S']['ci95'])

    def test_zero_contrast_is_void_not_a_pass_and_ties_are_not_success(self):
        result = p9.analyze(self.a[None], self.a[None], self.a, self.a,
                            self.ta, self.ta, bootstrap=100)
        full = result['per_seed']['0']['windows']['full']
        self.assertTrue(np.isnan(full['contrast']['S']['mean']))
        self.assertEqual(full['contrast']['S']['valid_units'], 0)
        self.assertTrue(np.isnan(full['recognition']['raw']['mean']))
        tied = self.analyze(self.a[None], self.a[None])
        self.assertEqual(tied['per_seed']['0']['windows']['full']['recognition']['raw']['mean'], 0)
        json.dumps(p9.json_ready(result), allow_nan=False)

    def test_frame_zero_is_ineligible_and_bad_shapes_or_starts_fail(self):
        result = self.analyze(self.good_a[None], self.good_b[None])
        frames = result['per_seed']['0']['framewise']
        self.assertTrue(np.isnan(frames['S_per_unit'][:, 0]).all())
        self.assertTrue((frames['raw_eligible_points'][:, 0] == 0).all())
        for kind in ('start', 'nan', 'frames', 'duplicate_seeds'):
            pa, pb = self.good_a[None].copy(), self.good_b[None].copy()
            kwargs = {}
            if kind == 'start':
                pa[0, 0, 0, 0] += 1
            elif kind == 'nan':
                pa[0, 0, 1, 0] = np.nan
            elif kind == 'frames':
                pa, pb = pa[:, :, :-1], pb[:, :, :-1]
            else:
                pa, pb = np.repeat(pa, 2, 0), np.repeat(pb, 2, 0)
                kwargs['seeds'] = [0, 0]
            with self.assertRaises(ValueError, msg=kind):
                self.analyze(pa, pb, **kwargs)

    def test_direct_norms_and_first_leg_oracle_identity(self):
        result = self.analyze(self.good_a[None], self.good_b[None])
        d = self.a - self.b - (self.ta - self.tb).mean(axis=0)
        for name, (start, stop) in p9.WINDOWS.items():
            w = slice(start, stop + 1)
            for arm, predicted, true in (('ab', self.good_a, self.a), ('ba', self.good_b, self.b)):
                residual = (predicted[:, w] - true[:, w]).reshape(8, -1)
                expected = np.linalg.norm(residual, axis=1) / np.linalg.norm(true[:, w].reshape(8, -1), axis=1)
                actual = result['per_seed']['0']['windows'][name]['arms'][arm]
                np.testing.assert_allclose(actual['relative_l2']['per_unit'], expected)
                np.testing.assert_allclose(actual['rmse']['per_unit'], np.sqrt((residual**2).mean(axis=1)))
        expected_oracle = np.sqrt((d[:, 101:]**2).sum(axis=(1, 2)) / (d**2).sum(axis=(1, 2)))
        oracle = result['first_leg_oracle']['windows']['full']['contrast']['S']['per_unit']
        np.testing.assert_allclose(oracle, expected_oracle)
        margin = result['per_seed']['0']['windows']['full']['margin_S_vs_first_leg_oracle']
        np.testing.assert_allclose(margin['per_unit'], 0.02 - expected_oracle)

    def test_boundary_impulse_is_counted_once_and_default_is_deterministic(self):
        pa = self.a.copy()
        pa[:, 100] += 0.1
        result = self.analyze(pa[None], self.b[None])
        windows = result['per_seed']['0']['windows']
        self.assertGreater(windows['first_leg']['contrast']['S']['mean'], 0)
        self.assertGreater(windows['second_leg']['contrast']['S']['mean'], 0)
        self.assertEqual(windows['post_handoff']['contrast']['S']['mean'], 0)
        first = p9.analyze(self.good_a[None], self.good_b[None], self.a, self.b, self.ta, self.tb)
        second = p9.analyze(self.good_a[None], self.good_b[None], self.a, self.b, self.ta, self.tb)
        self.assertEqual(first['uncertainty']['bootstrap_replicates'], 9999)
        self.assertEqual(json.dumps(p9.json_ready(first)), json.dumps(p9.json_ready(second)))

    def test_shared_loader_uses_only_synthetic_capture(self):
        with tempfile.TemporaryDirectory(dir=p9.HERE) as directory:
            path = Path(directory) / 'composed_seed0.npz'
            np.savez(path, pred_ab=self.good_a, pred_ba=self.good_b)
            pa, pb, receipts = p9.load_predictions(path, [0])
            np.testing.assert_array_equal(pa[0], self.good_a)
            np.testing.assert_array_equal(pb[0], self.good_b)
            self.assertEqual(len(receipts), 1)
            with self.assertRaises(ValueError):
                p9.load_predictions(path, [1])

    def test_cli_does_not_load_inputs_without_attestation(self):
        with patch.object(p9, 'load_predictions', side_effect=AssertionError('unexpected read')):
            with patch.object(p9, 'load_truth', side_effect=AssertionError('unexpected read')):
                with self.assertRaises(SystemExit) as outcome:
                    p9.main(['--predictions', '/not-opened', '--truth', '/not-opened',
                             '--output', str(p9.HERE / 'not-written.json')])
                self.assertEqual(outcome.exception.code, 2)


MUTATION = False


if __name__ == '__main__':
    unittest.main()
