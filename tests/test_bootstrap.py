from __future__ import annotations

import unittest

from aethereval.bootstrap import bootstrap_mean


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_mean(self) -> None:
        stats = bootstrap_mean([0.0, 1.0, 1.0, 0.0], n_resamples=500, seed=0)
        self.assertAlmostEqual(stats["mean"], 0.5, places=6)
        self.assertGreaterEqual(stats["stderr"], 0.0)
        self.assertLessEqual(stats["ci_low"], stats["mean"])
        self.assertGreaterEqual(stats["ci_high"], stats["mean"])

    def test_bootstrap_empty(self) -> None:
        stats = bootstrap_mean([])
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["stderr"], 0.0)


if __name__ == "__main__":
    unittest.main()
