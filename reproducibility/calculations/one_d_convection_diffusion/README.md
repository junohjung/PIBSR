# 1-D convection-diffusion calculations

## Verification entry point

```bash
python3 audit_results.py
```

The audit reads only `../../results/one_d_convection_diffusion` by default. Use
`--dry-run` to list and validate the required files, or `--results-root` to
point at another copy of the same result layout.

## Authoritative lineage

| Reported family | Final calculation source in the revision workspace | Packaged evidence |
| --- | --- | --- |
| Strong-form beta selection and data-only comparison | `Revisions/CMAME_minor_revision_v12_corrected_1d_sweep_20260816/experiments/one_d_corrected_sweep_v2/run_selection.py`, `sweep_common.py`, `run_postselection.py` | `results/one_d_convection_diffusion/final_postselection/strong_selection_v2/` |
| Integral-balance gamma family and reference-excluded selection | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/one_d_integral_balance_v2/run_gamma_family.py`, `select_reference_excluded.py`, `run_augmented_ls_certificates.py` | `results/one_d_convection_diffusion/selection/` and `augmented_ls_certificate/` |
| Final coordinate-compatible postselection | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/one_d_integral_balance_v2_postselection_addendum_v1/run_postselection_coordinate_compatible.py` | `results/one_d_convection_diffusion/final_postselection/balance_addendum_v1/` |
| Gamma 10 reporting-only row | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/one_d_integral_balance_v2_gamma10_reporting_addendum_v1/run_gamma10_reporting_addendum.py` and `experiments/one_d_gamma10_table_extension_v1/generate_table_record.py` | `results/one_d_convection_diffusion/final_postselection/gamma10_reporting_v1/` |
| Analytical-series resolution check | `Revisions/CMAME_minor_revision_v12_corrected_1d_sweep_20260816/experiments/one_d_corrected_sweep_v2/run_postselection.py` | `generation/reference_resolution.py` and `results/one_d_convection_diffusion/reference_resolution/reference_resolution.json` |

The selected strong weight is beta = `3e-5`; the selected integral-balance
weight is gamma = `0.1`. Gamma `10` is deliberately retained as an ineligible,
reporting-only sensitivity point and did not reopen selection.

The audit verifies the complete eight-value gamma candidate set, the five
eligible and Pareto candidates, the selected point's data/balance coordinates,
the trace and PDE guards, and the independent operator agreement. It also
checks the sparse-QR certificates for gamma `0` and `0.1`, including objective,
stationarity, rank, coefficient agreement, objective gap, and QR backward
error. The two doubled-resolution analytical-reference values are read from
their compact calculation record and can be recomputed with:

```bash
python3 generation/reference_resolution.py
```

That recomputation requires NumPy; `--dry-run` is standard-library only.

## Superseded work

`one_d_corrected_sweep_v1`, the strict `1e-7` stagnated postselection attempt,
and `one_d_integral_balance_v1` are exploratory or superseded. The first frozen
V2 postselection could not evaluate the float32 PINN time coordinates; the
coordinate-compatible addendum above is the final source for reported values.

## Observation archive

The packaged 400-frame trajectory and `SELECTION_INPUT_MANIFEST.json` are the
authoritative observation input. A targeted search of the revision workspace
found no exact source file for the original 1-D observation generator, so no
reconstructed solver is presented as if it were the historical calculation.
The archive provenance records a coarse explicit forward-Euler solve on 15
spatial nodes with effective step `0.006`, followed by interpolation to the
endpoint-inclusive `60 x 400` observation grid.
