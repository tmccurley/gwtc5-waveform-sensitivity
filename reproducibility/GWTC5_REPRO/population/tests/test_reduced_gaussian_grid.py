"""Exact algebra and quadrature controls are not real-data recovery tests."""

import math
import unittest
import numpy as np
from scipy.special import logsumexp
from src.reduced_gaussian_grid import prior_grid, kernel_log_factors, normalized_grid, refinement_check


class ReducedGridTests(unittest.TestCase):
    def test_uniform_prior_measure(self):
        nodes, weights = prior_grid(16)
        summary, masses = normalized_grid(nodes, weights, np.zeros(len(nodes)))
        np.testing.assert_allclose(summary['mean'], [0., -.5], atol=1e-14)
        np.testing.assert_allclose(summary['sd'], [1/math.sqrt(3), math.sqrt(.75)], atol=1e-14)
        self.assertAlmostEqual(summary['log_normalization'], 0)
        self.assertAlmostEqual(masses.sum(), 1)

    def test_kernel_chunks_match_direct_tails_and_support(self):
        x = np.array([-.99, -.5, 0., .3, .9])
        logs = np.array([2., -np.inf, -.1, 0., -1.])
        settings = np.array([[-1., -2.], [0., -.7], [1., -2.], [0., 1.]])
        expected = logsumexp(logs[None, :]-.5*((x[None, :]-settings[:, 0, None])/10**settings[:, 1, None])**2, axis=1)-math.log(5)
        for rows, batch in [(1, 1), (2, 3), (100, 100)]:
            actual = kernel_log_factors(x, logs, settings, 5, row_chunk=rows, batch_size=batch)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_injection_denominator_not_recovered_count(self):
        value = kernel_log_factors(np.zeros(4), np.zeros(4), [[0., 0.]], 100)
        self.assertAlmostEqual(value[0], math.log(.04))

    def test_common_normalization_cancels(self):
        events, selection, c = np.array([-2., -3., -4.]), -7., 1000.
        self.assertAlmostEqual((events+c).sum()-3*(selection+c), events.sum()-3*selection)

    def test_invalid_and_all_zero_factors_fail(self):
        for b in ([np.nan]*4, [np.inf]*4, [-np.inf]*4):
            with self.assertRaises((ValueError, ArithmeticError)):
                kernel_log_factors(np.zeros(4), b, [[0., 0.]], 4)
        with self.assertRaises(ValueError):
            kernel_log_factors(np.zeros(4), np.zeros(4), [[0., -3.]], 4)

    def test_invalid_posterior_fails(self):
        nodes, weights = prior_grid(8)
        for logs in (np.full(64, -np.inf), np.full(64, np.nan)):
            with self.assertRaises((ValueError, ArithmeticError)):
                normalized_grid(nodes, weights, logs)

    def test_refinement_flags_are_not_sampling_claims(self):
        a = dict(log_normalization=1., mean=[.1, -.7], sd=[.1, .2])
        self.assertTrue(refinement_check(a, a)['empirical_refinement_pass'])
        b = dict(a, mean=[.2, -.7])
        self.assertFalse(refinement_check(a, b)['empirical_refinement_pass'])
        self.assertFalse(refinement_check(a, a)['sampling_accuracy_validated'])


if __name__ == '__main__':
    unittest.main()
