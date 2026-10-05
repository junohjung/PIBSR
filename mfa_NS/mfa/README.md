# Lid-driven-cavity reviewer target

This directory contains the PI-BSR fitting executable used for the
lid-driven-cavity calculation and thirteen focused scientific tests. It
consumes the fixed cavity observations distributed under
`reproducibility/data/lid_driven_cavity/`; it does not contain or invoke the
Firedrake program that originally generated those observations.

From the repository root:

```bash
reproducibility/scripts/build_and_test.sh cavity
python3 reproducibility/scripts/run_selected.py cavity --execute
```

The runner supplies the spline controls, model-selection roles, selected
weights, and initializer. Snapshot and boundary readers reject missing,
truncated, nonnumeric, nonfinite, or overlong inputs.

The MFA source remains under the terms in `COPYING`. CppOptLib and LBFGS++
notices are retained beside their required headers. DIY and Eigen are shared by
the three reviewer targets under the repository-level `third_party` directory.
