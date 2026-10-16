"""Known weighted second moments and no false parameter-error interpretation."""

import unittest
import numpy as np
from src.reduced_selection_reference_summary import reference_rms, reference_rms_refinement


class ReferenceSummaryTests(unittest.TestCase):
    def test_known_integral(self):
        actual = reference_rms([[1, 2], [3, 4]], [.25, .75])
        np.testing.assert_allclose(actual, [np.sqrt(7), np.sqrt(13)])

    def test_zero_and_refinement(self):
        np.testing.assert_array_equal(reference_rms(np.zeros((2, 2)), [.5, .5]), [0, 0])
        self.assertTrue(reference_rms_refinement([0, 0], [0, 0])['numerical_refinement_pass'])
        self.assertTrue(reference_rms_refinement([1, 2], [1.005, 2.01])['numerical_refinement_pass'])
        self.assertFalse(reference_rms_refinement([1, 2], [1.1, 2])['numerical_refinement_pass'])

    def test_invalid_inputs(self):
        for weights in ([.1, .1], [-.1, 1.1], [np.nan, .5]):
            with self.assertRaises(ValueError):
                reference_rms([[1, 2], [3, 4]], weights)


if __name__ == '__main__':
    unittest.main()
