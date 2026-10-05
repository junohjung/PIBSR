"""Evaluate the published PINN checkpoint diagnostics."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import cavity_core as core
from cavity_evaluation import evaluate_gl8

ROOT = Path(__file__).resolve().parent
SEEDS = (1101, 2202, 3303, 4404, 5505)
FIELDS = (
    "q4_spatial_rms_mass",
    "q4_spatial_rms_x_momentum",
    "q4_spatial_rms_y_momentum",
    "signed_q1_mass",
    "signed_q1_x_momentum",
    "signed_q1_y_momentum",
)


def verify_inputs():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest.items():
        if (
            name.startswith("cavity/")
            and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Input checksum mismatch: " + name)


def load_model(seed):
    checkpoint = torch.load(
        ROOT / f"cavity/seed_{seed}/PINN_CHECKPOINT.pt", weights_only=True, map_location="cpu"
    )
    model = core.MLP(checkpoint["layers"], checkpoint["bounds"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate",))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "work/pinn_cavity")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    verify_inputs()
    times = np.loadtxt(ROOT / "cavity/evaluation_times.txt")
    for seed in SEEDS if args.seed is None else (args.seed,):
        output = args.output_dir / args.mode / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=False)
        model, report = (load_model(seed), {"mode": "checkpoint_evaluation"})
        _, rows, audit = evaluate_gl8(model, times, output)
        write_table(output / "balance_histories.tsv", rows)
        with (ROOT / f"cavity/seed_{seed}/PINN_GL8_Q4_Q1_HISTORIES.tsv").open() as stream:
            archived = list(csv.DictReader(stream, delimiter="\t"))
        actual = np.array([[float(row[k]) for k in FIELDS] for row in rows])
        expected = np.array([[float(row[k]) for k in FIELDS] for row in archived])
        report["maximum_balance_difference_from_archive"] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        report["evaluation_seconds"] = audit["evaluation_seconds"]
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(seed, json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
