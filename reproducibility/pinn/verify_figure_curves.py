#!/usr/bin/env python3
"""Compare checkpoint-generated PINN histories with Figures 4, 7 and 10."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1101, 2202, 3303, 4404, 5505)


def rows(path, delimiter="\t"):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    args = parser.parse_args()
    report = {}
    one_d = [
        row
        for row in rows(ROOT / "figure_data/one_d/balance_histories.tsv")
        if row["series_id"] == "declared_subset_pinn"
    ]
    expected = np.array([float(row["q1_gl8_signed"]) for row in one_d])
    with np.load(
        args.work_dir / "pinn_one_d/evaluate/predictions.npz", allow_pickle=False
    ) as archive:
        actual = archive["mass_balance"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    report["figure4_maximum_absolute_difference"] = float(np.max(np.abs(actual - expected)))
    burger_seeds = [
        rows(args.work_dir / f"pinn_burgers/evaluate/seed_{seed}/balance.csv", ",")
        for seed in SEEDS
    ]
    values = np.array(
        [
            [[float(row[key]) for key in ("u_balance", "v_balance")] for row in seed]
            for seed in burger_seeds
        ]
    )
    expected_rows = [
        row
        for row in rows(ROOT / "figure_data/two_d_burgers/balance_histories.tsv")
        if row["series_id"] == "pinn_mean"
    ]
    expected = np.array(
        [
            [
                float(row[key])
                for key in ("signed_q1_u", "signed_q1_v", "sample_sd_u", "sample_sd_v")
            ]
            for row in expected_rows
        ]
    )
    actual = np.column_stack((values.mean(axis=0), values.std(axis=0, ddof=1)))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
    report["figure7_mean_and_sd_maximum_absolute_difference"] = float(
        np.max(np.abs(actual - expected))
    )
    fields = (
        "q4_spatial_rms_mass",
        "q4_spatial_rms_x_momentum",
        "q4_spatial_rms_y_momentum",
        "signed_q1_mass",
        "signed_q1_x_momentum",
        "signed_q1_y_momentum",
    )
    cavity_seeds = [
        rows(args.work_dir / f"pinn_cavity/evaluate/seed_{seed}/balance_histories.tsv")
        for seed in SEEDS
    ]
    values = np.array(
        [[[float(row[key]) for key in fields] for row in seed] for seed in cavity_seeds]
    )
    expected_rows = rows(
        ROOT
        / "results/lid_driven_cavity/postselection/ramped_cavity_whole_domain_q1_figure_v13/PINN_AGGREGATE_HISTORIES.tsv"
    )
    maximum = 0.0
    for quantile, suffix in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
        actual = np.quantile(values, quantile, axis=0, method="linear")
        expected = np.array(
            [[float(row[key + "_" + suffix]) for key in fields] for row in expected_rows]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    report["figure10_median_and_iqr_maximum_absolute_difference"] = maximum
    (args.work_dir / "pinn_figure_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
