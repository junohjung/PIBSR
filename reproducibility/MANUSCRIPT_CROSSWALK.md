# Manuscript-to-package crosswalk

This crosswalk maps the calculation-bearing figures and tables in the revised
article and supplement to the released evidence. It uses LaTeX labels because
figure and table numbers can change during typesetting. Strings such as
`PIBSR_manuscript_v2.tex:891-922` elsewhere in the metadata are submitted-source
identifiers, not links to files in this repository; the TeX sources and
rendered manuscript figures are not duplicated here.

The scope terms below are deliberately narrow:

- **Audited result** means a case-level `audit_results.py` checks the reported
  numbers against the packaged campaign records. It does not mean that the
  optimizer was rerun.
- **Artifact retained** means the inputs, coefficients, or figure-ready tables
  are present, but the exact manuscript plotting program or complete upstream
  campaign is not necessarily included.
- **Context only** identifies a PINN or performance comparison that did not
  enter PI-BSR selection. The package may retain plotted values without the
  full training or benchmark campaign.
- **Not mirrored** means that the manuscript is the reporting record for that
  sweep or benchmark; the reproducibility package retains only the selected setting
  or adjacent evidence.

## Revised article

| Manuscript item | Packaged evidence | Released scope |
|---|---|---|
| `fig:pibsr_workflow` (PI-BSR workflow) | Scientific contracts in [`configs/`](configs/), role files in [`point_sets/`](point_sets/), and the selected-model runner [`scripts/run_selected.py`](scripts/run_selected.py) | **Method only.** The diagram explains the workflow rather than reporting a numerical result. |
| `tab:method_summary` (related-method comparison) | None required | **Method only.** This is a literature comparison, not an output of the numerical campaigns. |
| `fig:1dconvdiff_Case1` (1D observations and analytical reference) | Released trajectory and provenance in [`data/one_d_convection_diffusion/`](data/one_d_convection_diffusion/); analytical calculation in [`reference_resolution.py`](calculations/one_d_convection_diffusion/generation/reference_resolution.py); figure renderer in [`make_figures.py`](scripts/make_figures.py) | **Released-data figure reproducible.** The 400 stored frames use reconstruction times `0.4*j/399`, while their historical source times were `0.001*j`. The dashed curve is the continuous idealization, not the first fitting snapshot. This reproduces the current data contract, not necessarily historical artwork generated from 401 frames; final Illustrator typography is not reproduced. |
| `fig:1d_strong_weight_selection` and `tab:heldout_1d` | Full strong-sweep coordinates and sealed Pareto geometry in [`figure_data/one_d/`](figure_data/one_d/); hybrid records in [`results/one_d_convection_diffusion/selection/`](results/one_d_convection_diffusion/selection/); case audit and figure renderer | **Audited result and figure reproducible.** The selected beta, selected gamma, reported errors, and balance values are checked; the two-panel selection figure is regenerated from the released 26-candidate table and Pareto coordinates. |
| `fig:burgers_discrepancy` (forced observations versus unforced reference) | Coarse trajectories in [`data/two_d_burgers/observed/`](data/two_d_burgers/observed/) and [`data/two_d_burgers/reference/`](data/two_d_burgers/reference/); observation-generator lineage under [`calculations/two_d_burgers/`](calculations/two_d_burgers/); figure renderer | **Figure reproducible from retained data.** The five reported times, both velocity components, and absolute discrepancies are plotted directly from the two released trajectory families. |
| `fig:burgers_weight_selection` | Beta, kappa, and gamma plotting tables in [`figure_data/two_d_burgers/`](figure_data/two_d_burgers/), together with the selection seals and disjoint Design-B confirmation records | **Selected decisions audited and figure reproducible.** The complete plotted beta family, trace screen, Design-A selection ratios, and selected Design-B confirmation point are retained. The optimizer's 140/60 balance roles remain separate from the 200-time Design A and disjoint 200-time Design B. |
| `tab:burgers_offgrid_mechanics` and `fig:burgers_whole_domain_balance` | Post-selection tables under [`results/two_d_burgers/postselection/`](results/two_d_burgers/postselection/), consolidated figure histories in [`figure_data/two_d_burgers/balance_histories.tsv`](figure_data/two_d_burgers/balance_histories.tsv), the Burgers result audit, and figure renderer | **Audited result and figure reproducible, with a contextual exception.** Deterministic PI-BSR/MFA values and curves are retained. The optional `pinn/` extension includes five checkpoints and native evaluation; regenerated mean and sample-standard-deviation curves agree with the archived figure histories. |
| `fig:cavity_comparison` (coarse versus higher-resolution flow) | Actual pressure and reference frames in `figure_data/cavity_comparison/`; `scripts/make_figure8.py` | **Figure and metrics reproducible from fixed data.** All five displayed times are supplied. The inactive fitting pressure placeholder is not used. Firedrake remains excluded. |
| `fig:cavity_weight_selection` | Full family record in [`results/lid_driven_cavity/family_selection_v4.json`](results/lid_driven_cavity/family_selection_v4.json), validation metrics, plotted coordinates in [`CAVITY_FAMILY_SELECTION_COORDINATES.tsv`](results/lid_driven_cavity/postselection/ramped_cavity_weight_selection_figure_v2/CAVITY_FAMILY_SELECTION_COORDINATES.tsv), and figure renderer | **Selected outcomes audited; full matrix retained; figure reproducible.** The audit checks the frozen 4-by-4 grid, winners, and representative rejections. The selected hybrid has total gamma 0.04, split as `(gamma_S, gamma_I) = (0.02, 0.02)`. |
| `tab:cavity_balance_summary` and `fig:cavity_balance_histories` | GL8 summaries and rows under [`results/lid_driven_cavity/`](results/lid_driven_cavity/); deterministic and contextual curve tables in [`ramped_cavity_whole_domain_q1_figure_v13/`](results/lid_driven_cavity/postselection/ramped_cavity_whole_domain_q1_figure_v13/); figure renderer | **Audited result and figure reproducible** for the deterministic models. The optional `pinn/` extension includes five checkpoints and evaluation code; checkpoint GL8 histories and median/IQR curves reproduce the archived values. |
| `tab:efficiency_summary` (stored-value counts) | Grid and spline dimensions in [`configs/`](configs/), packaged input shapes, and coefficient counts in [`checkpoints/manifest.json`](checkpoints/manifest.json) | **Traceable arithmetic.** The table counts scalar samples and coefficients, not bytes. It is not a timing or compression-code benchmark. |
| `fig:selected_reconstruction_overview` and `fig:spacetime_volume_slices` | Selected coefficients, decoded grids and relocated ParaView states; commands in `EXTENDED_FIGURES.md` | **Fields, rendering and composition reproducible.** Independent spline decoding checks both volume grids. Figure 12 composition is not pixel-identical to typesetter artwork. |
| `tab:supp_1d_resolution`, `tab:supp_burgers_control_resolution`, and `tab:supp_cavity_control_resolution` (appendix control-lattice studies) | The selected lattices are fixed in [`configs/`](configs/) and their selected-model coefficients are in [`checkpoints/`](checkpoints/) | **Not mirrored.** The alternative-lattice campaign outputs are not in the curated archive, so these three sweeps cannot be replayed from a packaged table extract. |
| `tab:app_cavity_flow_discretization` | Fixed coarse/reference comparison frames and `scripts/make_figure8.py` | **Metrics recomputable.** The five-time, three-field comparison is recalculated, including independent pressure mean alignment. The full flow solve is intentionally excluded. |
| Unnumbered serial decoding and derivative-timing paragraph following `tab:efficiency_summary` | Source code for the reported evaluators is present in the three MFA trees | **Context only; not mirrored.** The fixed-work benchmark harness, raw repeats, and machine record underlying those timings are not part of this package. |

## Supplementary material

| Supplement item | Packaged evidence | Released scope |
|---|---|---|
| `tab:supp_information_roles` | All three [`configs/`](configs/), deterministic [`point_sets/`](point_sets/), cavity [`reporting_plan.npz`](data/lid_driven_cavity/reporting_plan.npz), and [`validate_package.py`](validate_package.py) | **Audited contract.** Counts and relevant disjointness checks are verified. For Burgers this includes the distinct 140/60 optimizer roles, Design A, and Design B; for the cavity the authoritative optimizer split is 700/300, not the reporting plan. |
| `tab:supp_1d_provenance` | [`data/one_d_convection_diffusion/SELECTION_INPUT_MANIFEST.json`](data/one_d_convection_diffusion/SELECTION_INPUT_MANIFEST.json), released observations, and the analytical-reference calculation | **Partial artifact.** Observation and reference roles are documented and hash sealed; the historical observation generator is not recovered. Contextual PINN checkpoint evaluation is included in `pinn/`. |
| `tab:supp_1d_components`, `tab:supp_1d_postselection`, and `tab:supp_1d_traces` | [`results/one_d_convection_diffusion/final_postselection/strong_selection_v2/`](results/one_d_convection_diffusion/final_postselection/strong_selection_v2/) and the 1D reported-metric audit | **Core metrics audited; remaining cells retained.** The audit checks the selected weights and primary midpoint, analytical, and whole-domain-balance comparisons. The archived summary contains the additional objective, GL2, and trace cells, but the audit does not extract every cell in these three tables. |
| `tab:supp_1d_gamma_family` | Candidate records, selection seal, and trajectories in [`results/one_d_convection_diffusion/selection/`](results/one_d_convection_diffusion/selection/), with independent sparse-QR evidence in [`augmented_ls_certificate/`](results/one_d_convection_diffusion/augmented_ls_certificate/) | **Selection audited; full table retained.** The audit checks the candidate/eligible sets, guards, selected gamma 0.1, selected ratios, and the gamma-10 reporting row. It does not independently extract every numerical cell for all eight candidates. |
| `fig:supp_1d_pinn` | [`pinn/`](pinn/README.md) | **Context only; partial figure coverage.** The checkpoint, fixed inputs, and native evaluation are included. The exact supplementary panel renderer is absent. Original checkpoint metrics reproduce; training from initialization is not included. |
| `tab:supp_burgers_reference_refinement` | [`results/two_d_burgers/reference_refinement/`](results/two_d_burgers/reference_refinement/) and [`calculations/two_d_burgers/generation/reference_refinement.py`](calculations/two_d_burgers/generation/reference_refinement.py) | **Audited result.** The archived refinement values are checked; the optional script reruns six complete reference solves and is outside lightweight verification. |
| `tab:supp_burgers_selection` | Beta, kappa, Design-A selection, and Design-B confirmation records listed for `fig:burgers_weight_selection` | **Audited result.** The separate balance-evaluation row is Design B; it is not the optimizer's 60-time internal validation role. |
| `fig:supp_burgers_postselection`, `fig:supp_burgers_selected_output_maps`, and `tab:supp_burgers_postselection` | Post-selection tables in [`results/two_d_burgers/postselection/`](results/two_d_burgers/postselection/) and selected/matched coefficients in [`checkpoints/two_d_burgers/`](checkpoints/two_d_burgers/) | **Core metrics audited; remaining cells retained.** The audit covers the main velocity, vorticity, strong-residual, trajectory, q=1, and final-role comparisons, but not every component and normalized-maximum cell in the supplementary table. The canonical kappa=1 mechanics rows are retained; the exact selected-output map renderer is absent. |
| `tab:supp_burgers_reference_sensitivity` | [`results/two_d_burgers/postselection/POSTSELECTION_SUMMARY.json`](results/two_d_burgers/postselection/POSTSELECTION_SUMMARY.json) and the Burgers result audit | **Audited result.** |
| `tab:supp_selector_stability` | 1D perturbations in [`strong_selection_v2/selection.json`](results/one_d_convection_diffusion/final_postselection/strong_selection_v2/selection.json); Burgers selected beta and family seal in [`BETA_SELECTION_SEAL.json`](results/two_d_burgers/selection/BETA_SELECTION_SEAL.json) | **Partial artifact.** The 1D sensitivity record is retained. The complete Burgers leave-one-out/transformation calculation is not separately packaged. |
| `tab:supp_cavity_candidates` | [`family_selection_v4.json`](results/lid_driven_cavity/family_selection_v4.json), [`validation_only_metrics_v4.json`](results/lid_driven_cavity/validation_only_metrics_v4.json), [`normalization_v4.json`](results/lid_driven_cavity/normalization_v4.json), and the cavity audit | **Full matrix retained; selected outcomes audited.** All 16 classifications are present. The audit checks the grid, winners, representative rejection reasons, and Stage-A normalizers rather than replaying all 16 fits. |
| `tab:supp_cavity_selection`, `tab:supp_cavity_shifted_q4`, and `tab:supp_cavity_q1` | Protected selected-family records and matched-control GL8 rows/summaries under [`results/lid_driven_cavity/`](results/lid_driven_cavity/) | **Audited result.** Canonical q=4, shifted q=4, and nested whole-domain q=1 values are checked. |
| `tab:supp_cavity_form_discrepancy` | [`RAW_FORM_CONSISTENCY.npz`](results/lid_driven_cavity/postselection/ramped_cavity_advective_conservative_consistency_v1/RAW_FORM_CONSISTENCY.npz), archived summaries, and [`calculations/lid_driven_cavity/postprocess_raw.py`](calculations/lid_driven_cavity/postprocess_raw.py) | **Audited and postprocessable.** NumPy is required to recompute the metrics from raw arrays. |
| `tab:supp_cavity_pinn_protocol` | Figure-ready aggregate and per-seed histories in [`ramped_cavity_whole_domain_q1_figure_v13/`](results/lid_driven_cavity/postselection/ramped_cavity_whole_domain_q1_figure_v13/) | **Context only.** All five checkpoints, fixed evaluation inputs, and GL8/GL12 evaluation are included in `pinn/`. Training from initialization is not included. |
| `tab:supp_pinn_protocol`, `tab:supp_pinn_fields`, and `tab:supp_pinn_residuals` (Burgers PINN) | Deterministic MFA/PI-BSR comparison values are present in [`results/two_d_burgers/postselection/`](results/two_d_burgers/postselection/) | **Context only; partial supplementary coverage.** The `pinn/` extension includes five checkpoints, fixed evaluation inputs, and native-grid metrics. Supplementary sealed-test and fully off-node PINN diagnostics are not replayed by this extension. |
| `tab:supp_pressure_rank_q` | q=4 rank contract and q-range declaration in [`configs/lid_driven_cavity.json`](configs/lid_driven_cavity.json); released pressure/balance operators and focused C++ tests | **Partial verification.** Package validation checks the active q=4 rank/nullity and strong-block rank. The complete q=1 through q=20 rank/conditioning table does not have a standalone packaged calculation script. |
| `tab:supp_cavity_pressure_diagnostics` | [`RAW_PRESSURE_DIAGNOSTICS.npz`](results/lid_driven_cavity/postselection/ramped_cavity_pressure_reference_diagnostics_v1/RAW_PRESSURE_DIAGNOSTICS.npz), per-time tables and summary in the same directory, and `postprocess_raw.py` | **Audited and postprocessable.** The higher-resolution pressure is a numerical comparison and was not used for selection. |
| `tab:supp_gradient_checks` | Historical values under `verification` in [`expected_results.json`](expected_results.json), archived source records and audit in [`calculations/verification/`](calculations/verification/), and focused C++ operator tests invoked by [`scripts/build_and_test.sh`](scripts/build_and_test.sh) | **Archived values audited.** The compact audit reconstructs five full-precision checks from hash-verified records. The release tests exercise current residual, boundary, balance, and derivative interfaces; they do not rerun every historical directional/Taylor campaign exactly as originally scheduled. |
| `tab:supp_cavity_optimization_records` | Loss histories and run manifests in [`strong_only__gamma_0p02/`](results/lid_driven_cavity/strong_only__gamma_0p02/) and [`gl4_hybrid__gamma_0p04/`](results/lid_driven_cavity/gl4_hybrid__gamma_0p04/) | **Archived single-run evidence.** Updates, final projected gradients, and wall times come from those completed runs; they are not repeat-timing benchmarks. |
| `tab:supp_setup_screen` and `tab:supp_collocation_scaling` | Selected grids and row counts remain in [`configs/`](configs/) and [`point_sets/`](point_sets/) | **Context only; not mirrored.** The benchmark harness, alternative setup outputs, raw repeats, RSS records, and full timing tables are not packaged. |

## How to check the mapped results

From the repository root, first run the package-wide validator and all three
case audits:

```sh
./reproducibility/scripts/verify_package.sh
```

The case audits can also be inspected independently:

```sh
python3 reproducibility/calculations/one_d_convection_diffusion/audit_results.py
python3 reproducibility/calculations/two_d_burgers/audit_results.py
python3 reproducibility/calculations/lid_driven_cavity/audit_results.py
python3 reproducibility/calculations/verification/audit_results.py
```

For the cavity raw-array calculations, check the inputs without NumPy or run
the full postprocessor in an environment with NumPy:

```sh
python3 reproducibility/calculations/lid_driven_cavity/postprocess_raw.py --dry-run
python3 reproducibility/calculations/lid_driven_cavity/postprocess_raw.py
```

With NumPy and Matplotlib installed, regenerate the eight supported main-text
numerical figures:

```sh
python3 reproducibility/scripts/make_figures.py all
```

See [`FIGURES.md`](FIGURES.md) for individual commands and the explicit limits.

No full PI-BSR selected-model refit, complete candidate sweep, or high-resolution cavity solve was performed while assembling this package. The contextual PINN subset supports checkpoint evaluation only; see [`pinn/README.md`](pinn/README.md).
