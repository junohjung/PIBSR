# Ramped lid-driven-cavity calculations

## Verification checks

Run these commands from this directory:

```bash
python3 audit_results.py
python3 postprocess_raw.py --dry-run
python3 postprocess_raw.py
```

`audit_results.py` uses only the Python standard library. It checks the
full-precision values in `reported_metrics.json` against the compact archived
tables. `postprocess_raw.py` requires NumPy and independently recomputes the
reported velocity, off-node residual, advective/conservative, and pressure
diagnostics from the packaged raw NPZ arrays. Its dry run is standard-library
only.

## Authoritative lineage

| Reported family | Final calculation source in the revision workspace | Packaged evidence |
| --- | --- | --- |
| GL4 within-family selection and Stage-A normalizers | `Revisions/CMAME_minor_revision_v8_ramped_only_20260805/experiments/ramped_cavity_gl4_v4/` | `results/lid_driven_cavity/{family_selection_v4.json,normalization_v4.json}` |
| Protected Stage-A, selected-hybrid, and selected-strong GL8 evaluation | `Revisions/CMAME_minor_revision_v8_ramped_only_20260805/experiments/ramped_cavity_gl4_v4_protected/` | `results/lid_driven_cavity/protected_gl8_v4/` |
| Stage-A velocity and off-node-strong comparison | `Revisions/CMAME_minor_revision_v8_ramped_only_20260805/results/ramped_cavity_gl4_v4/post_failure_diagnostic_v4_20260805/post_failure_diagnostic_v4.json` | byte-identical `results/lid_driven_cavity/stage_a_comparison_v4/post_failure_diagnostic_v4.json` |
| Matched fixed-strong-weight control and GL8 reporting | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_matched_strong_control_gl8_reporting_v1/` | `results/lid_driven_cavity/postselection/ramped_cavity_matched_strong_control_gl8_reporting_v1/` |
| Whole-domain Q1 histories | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_whole_domain_q1_figure_v13/` | `results/lid_driven_cavity/postselection/ramped_cavity_whole_domain_q1_figure_v13/` |
| Velocity and off-node strong diagnostics | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_matched_strong_field_diagnostics_v1/` | `results/lid_driven_cavity/postselection/ramped_cavity_matched_strong_field_diagnostics_v1/` |
| Advective/conservative residual forms | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_advective_conservative_consistency_v1/` | `results/lid_driven_cavity/postselection/ramped_cavity_advective_conservative_consistency_v1/` |
| Pressure and pressure-gradient diagnostics | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_pressure_reference_diagnostics_v1/` | `results/lid_driven_cavity/postselection/ramped_cavity_pressure_reference_diagnostics_v1/` |
| Supplementary weight-selection figure | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/ramped_cavity_weight_selection_figure_v2/` | `results/lid_driven_cavity/postselection/ramped_cavity_weight_selection_figure_v2/` |

The selected GL4 hybrid has total family weight `gamma = 0.04`, split as
`(gamma_S, gamma_I) = (0.02, 0.02)`. The
matched control has the same frozen strong weights as the hybrid and
`gamma_I = 0`; it is therefore distinct from the independently selected
strong-only endpoint at `gamma_S = 0.03`.

The audit covers canonical Q4, shifted Q4, and whole-domain Q1 GL8 balance;
the Stage-A, matched-control, selected-hybrid, and selected-strong roles;
within-family selection guards; velocity and fully off-node strong residuals;
advective/conservative identities; pressure diagnostics; and the archived
Stage-A normalizers and selected weights. The Stage-A comparison record also
anchors the main-table hybrid changes in off-node strong residuals
(`-23.6%`, `-32.6%`, and `-0.2%` after manuscript rounding).

## Flow-data boundary

The reproducibility package contains the generated cavity observations, dynamic
boundary records, point sets, and their hash manifests. The Firedrake solver,
its environment, and input-generation scripts are intentionally not
distributed. Generation settings and the hashes of the omitted historical
sources remain in the data manifests as provenance; they are not executable
package dependencies.

## Superseded work

The older impulsive-lid cavity calculations and whole-domain-Q1 versions
before V13 are not sources for the current figures or tables. The protected
V4 diagnostic is authoritative only for the Stage-A/selected-family
comparison it records. The later V14 matched-control, field, form, and
pressure calculations are authoritative for their respective comparisons.
