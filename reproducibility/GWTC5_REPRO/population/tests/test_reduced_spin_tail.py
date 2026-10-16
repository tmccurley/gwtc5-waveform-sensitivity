"""Deterministic tail controls, separate from observed population inference."""

import unittest
import numpy as np
from src.reduced_physical_moments import physical_moments
from src.reduced_spin_tail import log_tail_kernel, adaptive_log_tail, tail_summary


class ReducedTailTests(unittest.TestCase):
    def test_adaptive_controls_and_deep_positive_tails(self):
        settings = np.array([(m, ell) for m in (-1., -.97, 0., .97, 1.) for ell in (-2., -.7, 1.)])
        norm = physical_moments(settings)[:, 0]
        reference = np.array([adaptive_log_tail(p) for p in settings])
        for order in (64, 128):
            actual = log_tail_kernel(settings, order=order)-np.log(norm)
            np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=0)
        self.assertLess(reference[6], -4000)
        self.assertTrue(np.isfinite(reference).all())

    def test_symmetry_and_full_support(self):
        p = np.array([[-.3, -.7], [.3, -.7], [0., -2.]])
        norm = physical_moments(p)[:, 0]
        tail = log_tail_kernel(p)-np.log(norm)
        self.assertAlmostEqual(tail[0], tail[1], places=10)
        np.testing.assert_allclose(log_tail_kernel(p, threshold=0)-np.log(norm), 0, atol=1e-10)
        self.assertTrue(np.isneginf(log_tail_kernel(p, threshold=.99)).all())

    def test_plugin_is_not_probability_clipping(self):
        report, penalty = tail_summary(np.log([.01, .02]), np.log([.1, 2.]), [.25, .75])
        np.testing.assert_allclose(penalty, 104*np.log1p([.1, .01]))
        self.assertEqual(report['empirical_selection_factor_above_one_nodes'], 1)
        self.assertFalse(report['certified_likelihood_error_bound'])
        self.assertAlmostEqual(report['posterior_mean_plugin_log_penalty'], .25*penalty[0]+.75*penalty[1])

    def test_log_summary_preserves_underflow(self):
        report, _ = tail_summary([-5000., -6000.], [-5001., -6000.], [.5, .5])
        self.assertTrue(np.isfinite(report['posterior_log_mean_tail_probability']))
        self.assertLess(report['posterior_log_mean_tail_probability'], -4999)

    def test_invalid_values_fail(self):
        for p in ([[2., 0.]], [[0., -3.]], [[float('nan'), 0.]]):
            with self.assertRaises(ValueError):
                log_tail_kernel(p)
        with self.assertRaises(ValueError):
            tail_summary([.1], [0.], [1.])
        with self.assertRaises(ValueError):
            tail_summary([-1.], [-2.], [.5])


if __name__ == '__main__':
    unittest.main()
