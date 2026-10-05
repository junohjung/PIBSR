# Two-dimensional Burgers reviewer target

This directory contains the PI-BSR executable used for the manuscript's
two-dimensional coupled Burgers calculation and three focused scientific
tests. The driver accepts only the packaged `Burger2D` data contract; unrelated
MFA examples and tests are not part of this release.

From the repository root:

```bash
reproducibility/scripts/build_and_test.sh burgers
python3 reproducibility/scripts/run_selected.py burgers --execute
```

The runner supplies the observation directory, spline controls, fixed role
files, selected weights, and initializer. Input snapshots and the static
boundary table are checked for exact row counts and finite numeric values.

The MFA source remains under the terms in `COPYING`. CppOptLib and LBFGS++
notices are retained beside their required headers. DIY and Eigen are shared by
the three reviewer targets under the repository-level `third_party` directory.
