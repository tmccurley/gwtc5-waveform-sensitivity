"""Guard checkpoint shapes, original row counts and reference support."""

import unittest
import numpy as np
from scripts.prepare_reduced_spin_catalog import KEYS, validate_arrays, local_path, numpy_compatibility


class ReferenceCheckpointTests(unittest.TestCase):
    def test_process_local_quadrature_alias(self):
        numpy_compatibility()
        self.assertIs(np.trapezoid, np.trapz)
        x = np.array([0., .1, .7, 1.])
        y = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]])
        reference = np.sum((x[1:]-x[:-1])*(y[:, 1:]+y[:, :-1])/2, axis=1)
        np.testing.assert_array_equal(np.trapezoid(y, x, axis=1), reference)

    def frame(self):
        arrays = {key: np.zeros(3) for key in KEYS}
        arrays['row_index'] = np.arange(3)
        arrays['mass_1_source'][:] = 20.
        arrays['mass_ratio'][:] = .8
        return arrays

    def test_target_zero_keeps_prior_checks_and_row_count(self):
        arrays = self.frame()
        arrays['log_mass_redshift'][1] = -np.inf
        validate_arrays(arrays, 3)
        arrays['log_pe_continuous'][1] = -np.inf
        with self.assertRaises(ValueError):
            validate_arrays(arrays, 3)

    def test_missing_rows_or_columns_fail(self):
        with self.assertRaises(ValueError):
            validate_arrays(self.frame(), 2)
        arrays = self.frame()
        del arrays['row_index']
        with self.assertRaises(ValueError):
            validate_arrays(arrays, 3)

    def test_reference_endpoint_and_quadrature_errors_fail(self):
        arrays = self.frame()
        arrays['chi_eff'][0] = .99
        with self.assertRaises(ValueError):
            validate_arrays(arrays, 3)
        arrays = self.frame()
        arrays['spin_reference_relative_quadrature_error'][0] = -1.
        with self.assertRaises(ValueError):
            validate_arrays(arrays, 3)

    def test_path_translation_retains_full_subpath(self):
        import sys
        path = local_path('C:\\Users\\tmark\\Downloads\\example.hdf5')
        if sys.platform != 'win32':
            self.assertEqual(str(path), '/mnt/c/Users/tmark/Downloads/example.hdf5')


if __name__ == '__main__':
    unittest.main()
