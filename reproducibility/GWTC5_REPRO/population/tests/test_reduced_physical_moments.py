"""Moment identities and invalid inputs, not scientific recovery tests."""

import unittest
import numpy as np
from src.reduced_physical_moments import physical_moments, summarize_physical, physical_refinement


class PhysicalMomentTests(unittest.TestCase):
    def test_symmetry_and_reflection(self):
        values = physical_moments([[0, -2], [0, 1], [.97, -2], [-.97, -2]])
        np.testing.assert_allclose(values[:2, 1], 0, atol=1e-14)
        np.testing.assert_allclose(values[2, [0, 2]], values[3, [0, 2]], rtol=1e-12)
        self.assertAlmostEqual(values[2, 1], -values[3, 1], places=12)

    def test_narrow_interior_gaussian(self):
        values = physical_moments([[.1, -2]])[0]
        self.assertAlmostEqual(values[1], .099, places=12)
        self.assertAlmostEqual(values[2], .0099, places=12)
        self.assertAlmostEqual(values[0], .01*np.sqrt(2*np.pi), places=12)

    def test_physical_scaling_and_chunks(self):
        settings = [[-.99, -2], [0, -.7], [1, -2], [0, 1]]
        small = physical_moments(settings, a_max=.5, tail_width=.025, batch_size=1)
        large = physical_moments(settings, a_max=1, tail_width=.05, batch_size=50)
        np.testing.assert_allclose(small[:, 0], large[:, 0], rtol=1e-13)
        np.testing.assert_allclose(2*small[:, 1:], large[:, 1:], rtol=1e-13, atol=1e-14)

    def test_predictive_total_variance(self):
        values = [[1, -.1, .2], [1, .1, .4]]
        result = summarize_physical(values, [.5, .5])
        np.testing.assert_allclose(result['posterior_expectations'], [0, .3], atol=1e-14)
        np.testing.assert_allclose(result['posterior_sds'], [.1, .1], atol=1e-14)
        self.assertAlmostEqual(result['predictive_sd'], np.sqrt(.11), places=12)
        self.assertNotAlmostEqual(result['predictive_sd'], .3, places=3)
        self.assertFalse(result['physical_quantiles_certified'])
        self.assertTrue(physical_refinement(result, result)['empirical_physical_refinement_pass'])

    def test_invalid_inputs(self):
        for settings in ([[np.nan, 0]], [[2, 0]], [[0, -3]], []):
            with self.assertRaises(ValueError):
                physical_moments(settings)
        for masses in ([-1, 2], [0, 0], [.1, .1]):
            with self.assertRaises(ValueError):
                summarize_physical([[1, -.1, .2], [1, .1, .4]], masses)
        with self.assertRaises(ArithmeticError):
            summarize_physical([[1, .1, .2], [1, .1, .2]], [.5, .5])


if __name__ == '__main__':
    unittest.main()
