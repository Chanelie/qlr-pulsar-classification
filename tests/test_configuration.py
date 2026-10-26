import math
from pathlib import Path
import unittest

from qlr.config import Config


class ConfigurationTests(unittest.TestCase):
    def test_recorded_main_grid_configuration(self):
        self.assertEqual(Config.DATA_SEED, 42)
        self.assertEqual(Config.TEST_SIZE, 0.20)
        self.assertEqual(Config.ANGLE_SCALE, math.pi / 4)
        self.assertEqual(Config.LAYER_CONFIGS, [1, 2, 3, 5, 10])
        self.assertEqual(Config.SEEDS, [0, 1, 2])

    def test_release_paths_are_repository_relative(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Config.ROOT_DIR, root)
        self.assertEqual(Config.DATA_PATH, (root / "data" / "HTRU2.csv").resolve())


if __name__ == "__main__":
    unittest.main()
