"""Protocol selection tests with known optima and deterministic ties."""
import unittest
import numpy as np
from src.evaluation import (mix_probabilities, select_fusion_weight,
                            per_sample_top_k, select_sample_top_k)


class PredictionSelectionTests(unittest.TestCase):
    def test_fusion_selects_dev_ranking(self):
        y = np.array([[1, 0], [0, 1]])
        good = np.array([[.9, .1], [.1, .9]])
        bad = 1 - good
        result = select_fusion_weight(y, good, bad, [0, .5, 1])
        self.assertEqual(result["weight"], 1)
        np.testing.assert_allclose(mix_probabilities(good, bad, 1), good)

    def test_fusion_ties_and_shape_validation(self):
        y = np.eye(2)
        self.assertEqual(select_fusion_weight(y, y, y, [1, .5, 0])["weight"], 0)
        with self.assertRaises(ValueError):
            mix_probabilities(y, y[:, :1], .5)
        with self.assertRaises(ValueError):
            mix_probabilities(y, y, 2)

    def test_top_k_is_per_person_and_stable(self):
        values = np.array([[.2, .8, .8], [.9, .3, .1]])
        np.testing.assert_array_equal(per_sample_top_k(values, 1), [[0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_equal(per_sample_top_k(values, 2).sum(1), [2, 2])
        with self.assertRaises(ValueError):
            per_sample_top_k(values, 4)

    def test_dev_k_selection_does_not_use_eval(self):
        y = np.array([[1, 0, 0], [0, 1, 0]])
        p = np.array([[.8, .1, .1], [.1, .8, .1]])
        self.assertEqual(select_sample_top_k(y, p, [1, 2, 3])["k"], 1)


if __name__ == "__main__":
    unittest.main()
