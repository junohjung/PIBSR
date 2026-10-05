# PINN checkpoint evaluation

For the contextual PINN comparisons, this archive provides pretrained
checkpoints and evaluation scripts to reproduce the reported diagnostics and
figure curves. Training from initialization is not included.

## Installation and execution

Use Python 3.12. Run these commands from the repository root:

```sh
python3.12 -m venv .venv-pinn
.venv-pinn/bin/python -m pip install -r reproducibility/pinn/requirements.txt
.venv-pinn/bin/python reproducibility/pinn/run_one_d.py evaluate
.venv-pinn/bin/python reproducibility/pinn/run_burgers.py evaluate
.venv-pinn/bin/python reproducibility/pinn/run_cavity.py evaluate
.venv-pinn/bin/python reproducibility/pinn/verify_figure_curves.py
.venv-pinn/bin/python reproducibility/pinn/check_cavity_quadrature.py
```

The two-dimensional evaluators process five seeds by default; `--seed 1101`
selects one. Outputs are written under `reproducibility/work/pinn_one_d`,
`pinn_burgers`, and `pinn_cavity`. Existing output directories are not
overwritten. Use `--output-dir` for a fresh location; the curve verifier
accepts `--work-dir` for an alternate common output root.

## Evaluation scope

- Figure 4: the original 1D checkpoint and native trapezoidal balance curve.
- Figure 7: five Burgers checkpoints, native 41-by-41-grid diagnostics,
  whole-domain balance curves, and their mean and sample standard deviation.
- Figure 10: five cavity checkpoints, conservative GL8 balances at 200 times,
  and their median and type-7 interquartile interval. GL12 provides a separate
  quadrature-order check at the same times.

The 1D evaluator also recomputes loss components at the original optimization
coordinates. Generating these coordinates is an evaluation operation, not
a training workflow. Numerical comparisons use explicit floating-point
tolerances; exact equality across platforms is not required.

These are distinct native PINN protocols, not a common-method benchmark.
The supplementary sealed-test and fully off-node Burgers PINN analyses and
the exact supplementary 1D panel layout are outside this evaluation subset.
No Firedrake installation or flow-data generation is required.
