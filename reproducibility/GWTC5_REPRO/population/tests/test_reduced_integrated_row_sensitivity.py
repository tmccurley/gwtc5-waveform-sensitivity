"""Assembly controls use distinguishable groups, routes, arms and row counts."""

import unittest
import numpy as np
from src.reduced_integrated_row_sensitivity import COLUMNS, assemble_catalog, quarter_residuals
from src.reduced_row_group_kernels import grouped_kernel_log_factors
from src.reduced_gaussian_grid import prior_grid, normalized_grid
from src.reduced_physical_moments import physical_moments, summarize_physical


class IntegratedRowTests(unittest.TestCase):
    def test_all_columns_and_unchanged_selection(self):
        data = np.arange(4*7*11*2, dtype=float).reshape(4, 7, 11, 2)/100
        arms, selection = ['ST', 'XPNR', 'ST', 'XPNR'], np.linspace(-3, -1, 7)
        actual = assemble_catalog(data, arms, selection, events_per_arm=2)
        for k, (group, route, arm) in enumerate(COLUMNS):
            gi, ri = k//4, (k % 4)//2
            expected = np.array([sum(data[p, n, gi, ri] for p in range(4) if arms[p] == arm)-2*selection[n] for n in range(7)])
            np.testing.assert_allclose(actual[:, k], expected, rtol=0, atol=1e-13)

    def test_actual_quarter_denominators_and_corruption(self):
        factors, counts = grouped_kernel_log_factors(np.linspace(-.9, .9, 19), np.linspace(-4, 2, 19), [[0, -.7], [-1, -2]])
        data = np.stack((factors, factors+3), axis=-1)
        self.assertLess(max(quarter_residuals(data, counts, 19)), 1e-10)
        corrupted = data.copy(); corrupted[:, 4, :] += .1
        with self.assertRaises(ArithmeticError):
            quarter_residuals(corrupted, counts, 19)
        wrong = counts.copy(); wrong[1] += 1
        with self.assertRaises(ValueError):
            quarter_residuals(data, wrong, 19)

    def test_uniform_group_posteriors_and_physical_functionals(self):
        settings, prior = prior_grid(16)
        factors = np.full((2, len(settings), 11, 2), -3.)
        catalogs = assemble_catalog(factors, ['ST', 'XPNR'], np.full(len(settings), -2.), events_per_arm=1)
        physical = physical_moments(settings)
        for col in range(44):
            result, masses = normalized_grid(settings, prior, catalogs[:, col])
            np.testing.assert_allclose(masses, prior, rtol=0, atol=1e-16)
            np.testing.assert_allclose(result['mean'], [0, -.5], atol=1e-14)
            summary = summarize_physical(physical, masses)
            self.assertAlmostEqual(summary['predictive_mean'], 0, places=13)
            self.assertGreater(summary['predictive_sd'], 0)

    def test_missing_event_and_nonfinite_stop(self):
        with self.assertRaises(ValueError):
            assemble_catalog(np.zeros((2, 1, 11, 2)), ['ST', 'XPNR'], [0])
        with self.assertRaises(ArithmeticError):
            assemble_catalog(np.full((2, 1, 11, 2), -np.inf), ['ST', 'XPNR'], [0], events_per_arm=1)


if __name__ == '__main__':
    unittest.main()
