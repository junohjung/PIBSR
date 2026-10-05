#!/usr/bin/env python3
"""Check the cavity PINN balances with GL12 at all 200 reporting times."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_cavity import ROOT, SEEDS, FIELDS, verify_inputs, load_model, write_table
from cavity_evaluation import evaluate_gl8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT.parent / "work/pinn_cavity/quadrature"
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    verify_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    times = np.loadtxt(ROOT / "cavity/evaluation_times.txt")
    report = {}
    for seed in SEEDS:
        _, rows, _ = evaluate_gl8(load_model(seed), times, args.output_dir, order=12)
        write_table(args.output_dir / f"seed_{seed}_gl12.tsv", rows)
        with (ROOT / f"cavity/seed_{seed}/PINN_GL8_Q4_Q1_HISTORIES.tsv").open() as stream:
            old = list(csv.DictReader(stream, delimiter="\t"))
        a = np.array([[float(row[k]) for k in FIELDS] for row in old])
        b = np.array([[float(row[k]) for k in FIELDS] for row in rows])
        curve_error = np.max(np.abs(a - b), axis=0) / np.maximum(np.max(np.abs(a), axis=0), 1e-30)
        rms_a, rms_b = np.sqrt(np.mean(a * a, axis=0)), np.sqrt(np.mean(b * b, axis=0))
        rms_error = np.abs(rms_a - rms_b) / np.maximum(rms_a, 1e-30)
        maximum = float(max(curve_error.max(), rms_error.max()))
        if maximum > 1e-5:
            raise ValueError(f"GL8/GL12 discrepancy above declared tolerance: {seed}: {maximum}")
        report[seed] = {
            "maximum_relative_difference": maximum,
            "maximum_absolute_difference": float(np.max(np.abs(a - b))),
        }
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
