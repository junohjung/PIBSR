# Reported-calculation layer

This directory maps the calculations behind the current manuscript and
supplement to the compact artifacts in `../results`, the model files in
`../checkpoints`, and the fixed evaluation designs in `../point_sets`.

Each case contains:

- `README.md`: authoritative lineage and superseded campaigns;
- `reported_metrics.json`: full-precision reported values with packaged source
  paths;
- `audit_results.py`: a read-only consistency check of those values against the
  archived result artifacts.

From `reproducibility/`, run:

```bash
python3 calculations/one_d_convection_diffusion/audit_results.py
python3 calculations/two_d_burgers/audit_results.py
python3 calculations/lid_driven_cavity/audit_results.py
python3 calculations/verification/audit_results.py
```

All four audit scripts use only the Python standard library. Add `--dry-run`
to inspect the required inputs without evaluating them. The cavity
`postprocess_raw.py` calculation additionally requires NumPy and recomputes the
velocity, residual-form, and pressure diagnostics from the packaged raw NPZ
arrays.

These scripts audit already-computed scientific results. They do not retrain
the MFA models. Full model decoding and derivative evaluation depend on the
compiled C++ evaluators in the repository; the checkpoint manifest records the
exact packaged model roles.
