"""Small exact weight calculations do not certify posterior integration."""

import math
import unittest
import numpy as np
from scripts.check_reduced_spin_feasibility import weight_summary


class WeightSummaryTests(unittest.TestCase):
    def test_original_denominator_includes_support_zero(self):
        result = weight_summary(np.array([0., 0., -np.inf]), 3)
        self.assertAlmostEqual(result['log_factor'], math.log(2/3))
        self.assertAlmostEqual(result['kish_weight_concentration'], 2.)
        self.assertEqual(result['support_zero_rows'], 1)

    def test_extreme_finite_tail_not_support_zero(self):
        result = weight_summary(np.array([-10000., -10001., -np.inf]), 3)
        self.assertEqual(result['positive_weight_rows'], 2)
        self.assertAlmostEqual(result['largest_weight_share'], 1/(1+math.exp(-1)))
        self.assertAlmostEqual(result['log_factor'], -10000+math.log1p(math.exp(-1))-math.log(3))

    def test_all_zero_and_invalid_weights_fail(self):
        for values in ([-np.inf], [np.nan], [np.inf]):
            with self.assertRaises((ValueError, ArithmeticError)):
                weight_summary(np.array(values), 1)

    def test_injection_denominator_is_not_recovered_count(self):
        result = weight_summary(np.log([1., 2., 3.]), 100)
        self.assertAlmostEqual(result['log_factor'], math.log(.06))
        self.assertAlmostEqual(result['kish_weight_concentration'], 36/14)
        self.assertFalse(result['integration_error_validated'])


if __name__ == '__main__':
    unittest.main()
