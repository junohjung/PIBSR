import csv
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_tsv(relative_path):
    with (PACKAGE_ROOT / relative_path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class FigureInputTests(unittest.TestCase):
    def test_one_dimensional_selection_inputs(self):
        candidates = read_tsv("figure_data/one_d/strong_weight_candidates.tsv")
        geometry = read_tsv("figure_data/one_d/strong_weight_selection.tsv")
        self.assertEqual(len(candidates), 26)
        self.assertEqual(len(geometry), 13)
        selected = [row for row in geometry if row["selected"] == "True"]
        self.assertEqual([float(row["beta"]) for row in selected], [3.0e-5])

    def test_balance_history_inputs(self):
        one_d = read_tsv("figure_data/one_d/balance_histories.tsv")
        burgers = read_tsv("figure_data/two_d_burgers/balance_histories.tsv")
        self.assertEqual(len(one_d), 2800)
        self.assertEqual(len({row["series_id"] for row in one_d}), 7)
        self.assertEqual(len(burgers), 1402)
        self.assertEqual(len({row["series_id"] for row in burgers}), 7)

    def test_burgers_selection_inputs(self):
        beta = read_tsv("figure_data/two_d_burgers/beta_selection.tsv")
        kappa = read_tsv("figure_data/two_d_burgers/kappa_selection.tsv")
        gamma = read_tsv("figure_data/two_d_burgers/gamma_selection.tsv")
        self.assertEqual((len(beta), len(kappa), len(gamma)), (10, 5, 4))
        self.assertEqual(
            [float(row["beta"]) for row in beta if row["selected"] == "True"],
            [0.01],
        )
        self.assertEqual(
            [float(row["gamma"]) for row in gamma if row["selected"] == "True"],
            [0.03],
        )


if __name__ == "__main__":
    unittest.main()
