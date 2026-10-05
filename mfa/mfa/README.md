# One-dimensional convection-diffusion reviewer target

This directory contains only the C++ target used for the manuscript's
one-dimensional convection-diffusion calculations and its two focused unit
tests. The packaged runner supplies the released trajectory, initial-condition,
boundary-condition, checkpoint, and validation-point files.

From the repository root:

```bash
reproducibility/scripts/build_and_test.sh one_d
python3 reproducibility/scripts/run_selected.py one_d --execute
```

The flow executable accepts `-f` for the data directory, repeated `-v` options
for the two control-grid dimensions, and `-q` for the spline degree. The
physics-informed weights and reviewer diagnostics are configured through the
environment variables set by `run_selected.py`.

The MFA source remains under its original BSD-style terms in `COPYING`.
CppOptLib's license is retained beside its required headers. DIY and Eigen are
shared by the three reviewer targets under the repository-level `third_party`
directory.
