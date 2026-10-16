"""Predictive identities and failure controls on explicitly synthetic arrays."""

import unittest
import numpy as np
from src.reduced_physical_moments import physical_moments
from src.reduced_predictive_density import predictive_curve, curve_moments, curve_checks, curve_refinement


class PredictiveTests(unittest.TestCase):
    def test_two_component_mixture_and_total_variance(self):
        p = np.array([[-.1, -.7], [.2, -.7]])
        v = physical_moments(p)
        x = np.linspace(-.99, .99, 4097)
        components = predictive_curve(p, v[:, 0], np.eye(2), x)
        mixture = predictive_curve(p, v[:, 0], np.array([[.3], [.7]]), x)
        np.testing.assert_allclose(mixture[:, 0], components @ [.3, .7], atol=1e-14)
        mean = np.array([.3, .7]) @ v[:, 1]
        sd = np.sqrt(np.array([.3, .7]) @ (v[:, 2]**2+(v[:, 1]-mean)**2))
        self.assertTrue(curve_checks(curve_moments(x, mixture), [mean], [sd])['numerical_moment_pass'])
        self.assertFalse(np.allclose(mixture[:, 0], components[:, 0]))

    def test_reflection_and_nested_grids(self):
        p = np.array([[-.2, -.7], [.2, -.7]])
        v = physical_moments(p)
        x = np.linspace(-.99, .99, 2049)
        y = np.linspace(-.99, .99, 4097)
        a = predictive_curve(p, v[:, 0], np.eye(2), x)
        b = predictive_curve(p, v[:, 0], np.eye(2), y)
        np.testing.assert_allclose(b[:, 0], b[::-1, 1], atol=1e-13)
        self.assertTrue(curve_refinement(x, a, y, b)['numerical_curve_refinement_pass'])
        np.testing.assert_array_equal(b[[0, -1]], 0)

    def test_no_curve_renormalization(self):
        x = np.linspace(-.99, .99, 4097)
        f = np.ones((len(x), 1))
        m = curve_moments(x, f)
        self.assertAlmostEqual(m['integrals'][0], 1.98)
        self.assertFalse(m['curve_renormalized'])
        self.assertFalse(curve_checks(m, [0], [.5])['numerical_moment_pass'])

    def test_invalid_support_and_mass_fail(self):
        p = np.array([[0., -.7]])
        j0 = physical_moments(p)[:, 0]
        with self.assertRaises(ValueError):
            predictive_curve(p, j0, [[.5]], np.linspace(-.99, .99, 2049))
        with self.assertRaises(ValueError):
            predictive_curve(p, j0, [[1]], np.linspace(-.5, .5, 2049))
        with self.assertRaises(ArithmeticError):
            curve_moments(np.linspace(-.99, .99, 2049), np.zeros((2049, 1)))


if __name__ == '__main__':
    unittest.main()
