import unittest

import numpy as np
import pandas as pd

from qlr.data import HTRU2Preprocessor, subset_indices_for


class DataTests(unittest.TestCase):
    def test_preprocessor_is_stratified_and_deterministic(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            rng.normal(size=(100, 8)), columns=[f"x{i}" for i in range(8)]
        )
        frame["class"] = np.repeat([0, 1], [80, 20])

        prep_a = HTRU2Preprocessor(test_size=0.2, random_state=42)
        prep_b = HTRU2Preprocessor(test_size=0.2, random_state=42)
        split_a = prep_a.preprocess(frame)
        split_b = prep_b.preprocess(frame)

        for left, right in zip(split_a, split_b):
            np.testing.assert_allclose(left, right)
        self.assertEqual(split_a[3].sum(), 4)

    def test_subset_indices_are_reproducible_and_stratified(self):
        X = np.zeros((100, 8))
        y = np.repeat([0, 1], [80, 20])
        first = subset_indices_for(50, 3, X, y)
        second = subset_indices_for(50, 3, X, y)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(y[first].sum(), 10)


if __name__ == "__main__":
    unittest.main()
