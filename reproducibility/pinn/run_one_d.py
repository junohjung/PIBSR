"""Evaluate the published PINN checkpoint diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import one_d_core as core

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate",))
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "work/pinn_one_d")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name, digest in json.loads((ROOT / "MANIFEST.json").read_text()).items():
        if (
            name.startswith("one_d/")
            and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Input checksum mismatch: " + name)
    output = args.output_dir / args.mode
    output.mkdir(parents=True, exist_ok=False)
    with np.load(ROOT / "one_d/fields.npz", allow_pickle=False) as archive:
        x, t, observed, reference = [
            archive[key] for key in ("x", "t", "rho_nonconservative", "rho_analytical")
        ]
    protocol = json.loads((ROOT / "one_d/protocol.json").read_text())
    core.set_seed(1234)
    model = core.MLP(
        [2, 64, 64, 64, 64, 64, 1],
        (float(x.min()), float(x.max())),
        (float(t.min()), float(t.max())),
    )
    archived = torch.load(ROOT / "one_d/checkpoint.pt", weights_only=True, map_location="cpu")
    report = {"mode": args.mode, "torch": str(torch.__version__), "numpy": np.__version__}
    elapsed = 0.0
    sets = core.sample_training_sets(x, t, observed, 8000, 12000, 1234)
    model.load_state_dict(archived)
    result = core.evaluate_model(
        model, x, t, observed, reference, 1.0, 0.05, elapsed, torch.device("cpu")
    )
    training_tensors = {
        name: core.to_tensor(
            getattr(sets, name), torch.device("cpu"), requires_grad=name in {"x_r", "t_r"}
        )
        for name in sets.__dataclass_fields__
    }
    components = core.compute_loss_components(model, training_tensors, a=1.0, nu=0.05)
    report["optimization_set_losses"] = {
        name: float(value.detach()) for name, value in components.items()
    }
    names = {
        "MSE_data": "MSE_data",
        "MSE_ref": "MSE_reference",
        "MSE_PDE": "MSE_pde",
        "MSE_IC": "MSE_ic",
        "MSE_BC": "MSE_bc",
        "Ebal_RMS": "mass_balance_RMS",
        "final_time_L2_ref": "final_time_L2_reference",
        "final_time_Linf_ref": "final_time_Linf_reference",
    }
    expected = protocol["full_60_by_400_grid_diagnostics"]
    report["metrics"] = {
        key: {"actual": result.metrics[key], "expected": expected[other]}
        for key, other in names.items()
    }
    for key, row in report["metrics"].items():
        np.testing.assert_allclose(
            row["actual"], row["expected"], rtol=2e-05, atol=1e-09, err_msg=key
        )
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "predictions.npz", x=x, t=t, rho=result.rho_pred, mass_balance=result.mass_balance
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
