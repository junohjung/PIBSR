# Reported results

`expected_results.json` is the package-wide machine-readable acceptance record.
The case-level `calculations/*/reported_metrics.json` files map those values to
the curated campaign artifacts, and the accompanying `audit_results.py`
programs check the extraction. Values printed only to manuscript precision are
labelled `manuscript_rounded`; archived full-precision values retain the label
`archived_full_precision` and their recorded comparison tolerances.

The `source` fields cite the submitted manuscript or supplementary TeX by
filename and line number. Those submission sources are not duplicated in this
code package; the stable case/metric keys, case-level extracts, and archived
calculation artifacts provide the self-contained numerical cross-reference.

## One-dimensional convection--diffusion

All models use the same cubic `40 x 40` coefficient lattice. The selected
strong-form weight is `beta = 3e-5`; the selected hybrid adds `gamma = 0.1`.

| Model | Midpoint data MSE | Midpoint PDE MSE | Analytical relative L2 | q=4 balance MSE | q=1 GL8 RMS |
|---|---:|---:|---:|---:|---:|
| Data-only MFA | 1.086e-6 | 1.862 | 2.535e-2 | -- | 1.196e-1 |
| Strong form | 5.204e-6 | 2.821e-1 | 2.330e-2 | 6.484e-3 | 6.673e-2 |
| Selected hybrid | 5.818e-6 | 2.510e-1 | 2.310e-2 | 4.617e-3 | 5.624e-2 |

Relative to data-only MFA, the strong model reduces midpoint PDE MSE by 84.8%
and analytical error by 8.1%. Relative to the strong model, the hybrid reduces
q=4 balance MSE by 28.8% and q=1 GL8 RMS by 15.7%, while midpoint data MSE
increases by 11.8%. The analytical reference is post-selection only.

Primary evidence is under
`results/one_d_convection_diffusion/selection/`, `final_postselection/`, and
`augmented_ls_certificate/`.

## Two-dimensional Burgers

The selected model is `(beta, kappa, gamma) = (0.01, 10, 0.03)`. Its matched
control has `gamma = 0`; both contain 54,000 coefficients.

| Quantity | Strong form | Selected hybrid | Reported change |
|---|---:|---:|---:|
| q=4 GL8 balance RMS, u | 0.03130 | 0.03092 | -1.20% |
| q=4 GL8 balance RMS, v | 0.02321 | 0.02263 | -2.51% |
| q=1 balance RMS on the 41-time mechanics design, u | 0.1134 | 0.1120 | -1.25% |
| q=1 balance RMS on the 41-time mechanics design, v | 0.05322 | 0.05154 | -3.15% |
| Off-node velocity relative L2 | 0.2427 | 0.2426 | -0.053% |
| Off-node vorticity relative L2 | 0.3423 | 0.3422 | -0.018% |

The supported conclusion is a modest improvement in consistency with the
prescribed unforced balance operator. Field and mechanics values are
numerically similar. The executable's integral block used its 140/60 internal
fit/validation split. Gamma candidates were ranked on the separate 200-time
sealed Design A; the q=4 GL8 values in this table come from the disjoint
200-time Design B confirmation.

Primary evidence is under
`results/two_d_burgers/selection/`, `kappa_screen/`,
`stage_b_interface_compatibility/`, `postselection/`, and
`reference_refinement/`.

## Lid-driven cavity

The primary comparison holds the strong budget fixed. The matched control is
`(gamma_S, gamma_I) = (0.02, 0)` and the selected GL4 hybrid is
`(0.02, 0.02)`.

| Quantity | Matched strong | Selected hybrid | Reported change |
|---|---:|---:|---:|
| Canonical q=4 mass RMS | 3.811e-3 | 3.659e-3 | -4.0% |
| Canonical q=4 x-momentum RMS | 1.096e-2 | 2.968e-3 | -72.9% |
| Canonical q=4 y-momentum RMS | 9.904e-3 | 7.796e-4 | -92.1% |
| Velocity-test u RMSE | 1.292e-4 | 1.351e-4 | +4.6% |
| Velocity-test v RMSE | 9.503e-5 | 9.622e-5 | +1.3% |
| Mean-aligned pressure relative L2 | 0.20777 | 0.16923 | -18.55% |

The regional momentum reduction persists on the shifted partition. Pressure
gradient behavior is mixed across evaluation scales, so only the mean-aligned
pressure-field improvement is claimed. The 700/300 campaign split was used for
fit and selection; the 200 sealed midpoint times and higher-resolution pressure
comparison are post-selection only.

Primary evidence is under `results/lid_driven_cavity/`, especially
`family_selection_v4.json`, the two selected run directories,
`protected_gl8_v4/`, and `postselection/`.

## Numerical verification checks

| Check | Recorded maximum discrepancy |
|---|---:|
| Burgers strong-objective directional derivative | 6.62e-11 relative |
| Cavity strong-objective directional derivative | 4.16e-12 relative |
| Burgers direct/flux balance closure | 1.891e-16 |
| Independent cavity balance implementation | 1.13e-16 absolute |
| Physical-coordinate derivative scaling | 6.003e-9 |

These values are stored under `verification` in `expected_results.json` with
their precision label, source location, and tolerance. The archived values are
checked against their compact source records by
`calculations/verification/audit_results.py`. The focused C++ tests exercise
the released residual, balance, checkpoint, boundary, and optimizer
interfaces; they do not rerun the original verification campaigns or full
fits.
