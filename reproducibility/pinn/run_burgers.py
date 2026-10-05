"""Evaluate the published PINN checkpoint diagnostics."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import burgers_core as core

ROOT = Path(__file__).resolve().parent
SEEDS = (1101, 2202, 3303, 4404, 5505)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate",))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "work/pinn_burgers")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for name, expected in json.loads((ROOT / "MANIFEST.json").read_text()).items():
        if (
            name.startswith("burgers/")
            and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Input checksum mismatch: " + name)
    data = ROOT / "burgers"
    config = json.loads((data / "config.json").read_text())
    for seed in SEEDS if args.seed is None else (args.seed,):
        output = args.output_dir / args.mode / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=False)
        core.set_seed(seed)
        model = core.MLP(
            config["layers"],
            ((0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
            config["ic_Au"],
            config["ic_Av"],
            True,
            config["hard_time_beta"],
        )
        archived = torch.load(
            data / f"seed_{seed}/pinn_burgers2d_model.pt", map_location="cpu", weights_only=True
        )
        report = {
            "mode": args.mode,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "threads": args.threads,
        }
        elapsed = 0.0
        model.load_state_dict(archived)
        with np.load(data / "evaluation_fields.npz", allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        result = core.evaluate_model(
            model,
            **fields,
            nu=0.01,
            train_time_s=elapsed,
            device=torch.device("cpu"),
            eval_batch=4096,
        )
        expected = json.loads((data / f"seed_{seed}/metrics.json").read_text())
        comparisons = {
            key: {"actual": value, "expected": expected[key]}
            for key, value in result.metrics.items()
            if key != "train_time_seconds"
        }
        for key, row in comparisons.items():
            np.testing.assert_allclose(
                row["actual"], row["expected"], rtol=2e-05, atol=1e-10, err_msg=key
            )
        report["metrics"] = comparisons
        with (output / "balance.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("t", "u_balance", "v_balance"))
            writer.writerows(zip(fields["t"], result.ru_mass, result.rv_mass))
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{seed}: {len(comparisons)} diagnostics recomputed; mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
