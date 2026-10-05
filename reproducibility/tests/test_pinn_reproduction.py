"""Small tests for the optional PINN reproduction extension."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1] / "pinn"
sys.path.insert(0, str(ROOT))


class PinnInputTests(unittest.TestCase):

    def test_manifest(self):
        for name, expected in json.loads((ROOT / "MANIFEST.json").read_text()).items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
    "Optional PINN dependencies unavailable",
)
class PinnOperatorTests(unittest.TestCase):

    def test_checkpoint_parameter_counts(self):
        import torch
        import cavity_core
        import burgers_core
        import one_d_core

        models = [
            (cavity_core.MLP([3] + [108] * 6 + [3], ((0.0, 1.0), (0.0, 1.0), (0.0, 10.0))), 59619),
            (burgers_core.MLP([3] + [103] * 6 + [2], ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))), 54180),
            (one_d_core.MLP([2] + [64] * 5 + [1], (0.0, 1.0), (0.0, 0.4)), 16897),
        ]
        for model, count in models:
            self.assertEqual(sum((p.numel() for p in model.parameters())), count)
            self.assertTrue(all((torch.isfinite(p).all() for p in model.parameters())))

    def test_quadrature_area(self):
        import numpy as np
        from cavity_evaluation import make_q4_quadrature

        for order in (8, 12):
            points, blocks = make_q4_quadrature(order)
            self.assertEqual(points.shape[0], 16 * (order**2 + 4 * order))
            self.assertAlmostEqual(
                sum((np.sum(b["volume_weights"]) for b in blocks)), 1.0, places=14
            )

    def test_burgers_hard_boundary_and_initial_condition(self):
        import torch
        from burgers_core import MLP

        model = MLP(
            [3, 8, 8, 2],
            ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
            0.8906784057617188,
            0.7969791293144226,
            True,
        )
        boundary = torch.tensor(
            [[0.0, 0.7, 1.0], [2.0, 0.7, 1.0], [0.7, 0.0, 1.0], [0.7, 2.0, 1.0]]
        )
        self.assertTrue(torch.equal(model(boundary), torch.zeros(4, 2)))
        points = torch.tensor([[0.5, 0.7, 0.0], [1.0, 1.0, 0.0]])
        expected = torch.cat(model._analytic_ic(points), dim=1)
        self.assertTrue(torch.equal(model(points), expected))


if __name__ == "__main__":
    unittest.main()
