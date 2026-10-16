"""Exact partition diagnostics do not establish integration precision."""

import math
import unittest
import numpy as np
from scripts.check_reduced_catalog_weights import fixed_settings, partition_summary


class CatalogWeightTests(unittest.TestCase):
    def test_fixed_settings(self):
        self.assertEqual(len(fixed_settings()), 13)
        self.assertEqual(fixed_settings()[9:], [(.1, -.7), (-1., -2.), (1., -2.), (0., 1.)])

    def test_constant_weights(self):
        result = partition_summary(np.full(19, -10000.))
        for route in ('contiguous', 'interleaved'):
            self.assertAlmostEqual(result[route]['log_average_span'], 0.)
            self.assertEqual(sum(g['rows'] for g in result[route]['groups']), 19)
        self.assertFalse(result['independent_replicates'])

    def test_support_zeros_and_original_group_denominator(self):
        result = partition_summary(np.array([0., -np.inf]*8))
        self.assertAlmostEqual(result['contiguous']['groups'][0]['log_factor'], math.log(.5))
        self.assertFalse(result['interleaved']['all_groups_positive'])
        self.assertIsNone(result['interleaved']['log_average_span'])
        self.assertIsNone(result['interleaved']['groups'][1]['log_factor'])

    def test_invalid_values_fail(self):
        for x in ([0., 0., 0., np.nan], [0., 0., 0., np.inf], [0.]*3):
            with self.assertRaises(ValueError):
                partition_summary(x)


if __name__ == '__main__':
    unittest.main()
