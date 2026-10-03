import os
import tempfile
import unittest
from src import preprocessing as pp


class TestPreprocessing(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(pp.mean([2, 4, 6, 8]), 5)

    def test_std(self):
        self.assertAlmostEqual(pp.std([2, 4, 6, 8]), 2.2360679, places=6)

    def test_min_max_scale(self):
        result = pp.min_max_scale([2, 4, 6, 8])
        for got, exp in zip(result, [0, 1/3, 2/3, 1]):
            self.assertAlmostEqual(got, exp)

    def test_z_score(self):
        z = pp.z_score([2, 4, 6, 8])
        self.assertAlmostEqual(pp.mean(z), 0)
        self.assertAlmostEqual(pp.std(z), 1)

    def test_train_test_split(self):
        train, test = pp.train_test_split(list(range(1, 11)), 0.2)
        self.assertEqual(test, [9, 10])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            pp.mean([])

    def test_load_csv_column(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("age,salary\n25,50000\n35,70000\n")
            path = f.name
        try:
            self.assertEqual(pp.load_csv_column(path, "salary"), [50000.0, 70000.0])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
