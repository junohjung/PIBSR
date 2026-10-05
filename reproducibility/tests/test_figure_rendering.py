"""Optional rendering regressions; install requirements-figures.txt to run."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_PLOTTING = all(importlib.util.find_spec(name) is not None for name in ("numpy", "matplotlib"))


@unittest.skipUnless(HAS_PLOTTING, "optional NumPy/Matplotlib dependencies not installed")
class FigureRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        spec = importlib.util.spec_from_file_location("figures", ROOT / "scripts/make_figures.py")
        cls.figures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.figures)
        cls.np, cls.plt = np, plt

    def capture(self, function):
        captured = []
        original = self.figures.save_figure
        self.figures.save_figure = lambda plt, figure, *args: captured.append(figure) or []
        try:
            with tempfile.TemporaryDirectory() as directory:
                function(self.plt, self.np, Path(directory))
            return captured[0]
        finally:
            self.figures.save_figure = original

    def tearDown(self):
        self.plt.close("all")

    def test_initial_curve_is_continuous_idealization(self):
        figure = self.capture(self.figures.one_d_observations)
        curve = figure.axes[0].lines[2]
        self.np.testing.assert_array_equal(curve.get_ydata(), self.figures.initial_condition(self.np, curve.get_xdata()))
        contours = [collection for axis in figure.axes for collection in axis.collections if hasattr(collection, "extend")]
        self.assertGreaterEqual(sum(collection.extend == "both" for collection in contours), 3)

    def test_regional_balance_lines_are_not_clipped(self):
        figure = self.capture(self.figures.one_d_balance_histories)
        axis = figure.axes[0]
        lower, upper = axis.get_ylim()
        for line in axis.lines:
            self.assertGreaterEqual(min(line.get_ydata()), lower)
            self.assertLessEqual(max(line.get_ydata()), upper)


if __name__ == "__main__":
    unittest.main()
