"""Covariance replay is a numerical control, not real-design validation."""

import unittest
import numpy as np
from src.reduced_selection_reference import replay_covariance_references
from src.sample_selection_audit import fixed_stratum_reference, log_ratio_reference_sd
from src.stopped_campaign_reference import stopped_stratum_reference


class ReducedSelectionReferenceTests(unittest.TestCase):
    def setUp(self):
        with np.errstate(divide='ignore'):
            self.logs = np.log([[1., 2., 1.], [3., 6., .5], [0., 0., 0.], [.2, .4, 4.], [2., 4., 1.]])
        self.strata = np.array([0, 0, 0, 1, 1])
        self.n, self.r = np.array([20, 30, 40]), np.array([5, 6, 4])

    def test_both_references_and_zero_stratum(self):
        replay = replay_covariance_references(self.logs, self.strata, self.n, self.r)
        fixed = fixed_stratum_reference(self.logs, self.strata, self.n)
        stopped = stopped_stratum_reference(self.logs, self.strata, self.n, self.r)
        np.testing.assert_allclose(replay['fixed_relative_covariance'], fixed['relative_covariance'], atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(replay['stopped_relative_covariance'], stopped['relative_covariance'], atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(replay['point_multipliers'], stopped['point_multiplier_vs_published'], atol=1e-14)
        np.testing.assert_allclose(replay['log_uncorrected_sums'], fixed['log_sums'], atol=1e-14)

    def test_column_shift_invariance_and_shared_ratio(self):
        a = replay_covariance_references(self.logs, self.strata, self.n, self.r)
        b = replay_covariance_references(self.logs+np.array([700., -700., 1000.]), self.strata, self.n, self.r)
        for key in ('fixed_relative_covariance', 'stopped_relative_covariance', 'point_multipliers'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-14)
        self.assertLess(log_ratio_reference_sd(a['fixed_relative_covariance'], 1, 0), 2e-8)

    def test_omitted_zero_rows(self):
        a = replay_covariance_references(self.logs, self.strata, self.n, self.r)
        keep = np.array([True, True, False, True, True])
        b = replay_covariance_references(self.logs[keep], self.strata[keep], self.n, self.r)
        for key in ('fixed_relative_covariance', 'stopped_relative_covariance', 'point_multipliers'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-14)

    def test_invalid_counts_and_unavailable_support(self):
        with self.assertRaises(ValueError):
            replay_covariance_references(self.logs, self.strata, self.n.astype(float), self.r)
        with self.assertRaises(ValueError):
            replay_covariance_references(self.logs, self.strata, self.n, np.array([2, 6, 4]))
        with self.assertRaises(ArithmeticError):
            replay_covariance_references(np.full((5, 1), -np.inf), self.strata, self.n, self.r)


if __name__ == '__main__':
    unittest.main()
