# 2-D Burgers calculations

## Verification entry point

```bash
python3 audit_results.py
```

The standard audit has no third-party dependency and recomputes the GL4/GL8
time summaries from the archived trajectories. `--dry-run` checks the fixed
input layout without comparing values.

## Authoritative lineage

| Reported family | Final calculation source in the revision workspace | Packaged evidence |
| --- | --- | --- |
| Strong-form beta selection | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/burgers_bc_corrected_v3_campaign_v1/run_beta_family.py`, `select_beta_reference_excluded.py` | `results/two_d_burgers/selection/BETA_SELECTION_SEAL.json` |
| Kappa screen | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/burgers_bc_corrected_v4_continuation_v1/screen_kappa.py` | `results/two_d_burgers/kappa_screen/KAPPA_SELECTION_SEAL.json` |
| Final Stage-A balance selection | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/burgers_bc_corrected_v5_stage_a_serialization_addendum_v2/select_stage_a_reference_excluded.py` | `results/two_d_burgers/selection/STAGE_A_SELECTION_SEAL.json` |
| Independent GL4/GL8 Stage-B evaluation | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/burgers_bc_corrected_v5_stage_b_interface_compatibility_continuation_v2/run_stage_b.py`, `continuation_common.py` | `results/two_d_burgers/stage_b_interface_compatibility/` |
| Off-node fields, strong residuals, mechanics, and final-role metrics | `Revisions/CMAME_minor_revision_v14_integral_balance_extension_20260819/experiments/burgers_bc_corrected_v5_postselection_access_authorization_v1/evaluate_postselection_v5.py` | `results/two_d_burgers/postselection/` |
| Reference refinement table | `Revisions/PIBSR_research_archive/scripts/common/burgers_reference_refinement.py` | `results/two_d_burgers/reference_refinement/burgers_reference_refinement.{json,tsv}` |

The final weights are beta = `0.01`, kappa = `10`, and gamma = `0.03`.
The earlier V4 selection seal chose gamma `0.01` under a different
validation-data epsilon-constraint; it is not the final Stage-A rule used by
the current Stage-B and postselection results. The V5 Stage-A
serialization-compatible selection above is authoritative.

Three temporal roles must not be conflated:

1. The optimizer used a fixed 140-time integral-balance fit role and a
   60-time integral-balance validation role. Their indices are packaged in
   `point_sets/two_d_burgers/integral_balance/`.
2. Final weight selection used a separate 200-time fully off-node Design A,
   `0.01*(i+1/3)`. Its archived trajectory, summary, and three-candidate local
   sensitivity are in `results/two_d_burgers/selection/stage_a_design_a/` and
   `STAGE_A_CLASSIFIED_CANDIDATES.json`.
3. The selected gamma was confirmed once on the disjoint 200-time Design B,
   `0.01*(i+2/3)`, with both GL4 and GL8. Those records are in
   `results/two_d_burgers/stage_b_interface_compatibility/`.

The standard audit checks all three roles, reconstructs both 200-time design
hashes from the archived trajectory rows, and verifies the gamma `0.01`,
`0.03`, and `0.1` local-sensitivity classification. It also checks the beta
selector's endpoint-chord distance and the maximum normalized trace plus gate
decision for every screened kappa (`1`, `3`, `10`, and `30`).
Reference-interpolation sensitivity is checked from
`postselection/POSTSELECTION_SUMMARY.json`.

## Data and forcing generation

`archived_source/generate_burgers_observed.py` is the byte-identical copy of
the published coarse-observation generator (SHA-256
`aca87d645aaa44577083437e617e36dc42d97a7d500e513b05da2d7d940e5fc3`).
It uses fourth-order centered differences, RK4, and the fixed smooth forcing.
The active `generation/generate_burgers_observed.py` is a portable safety
adaptation: it updates the package-layout description and refuses to write
into a nonempty output directory. For a new output directory:

```bash
python3 generation/generate_burgers_observed.py --output-dir /path/to/new-output
```

`generation/generate_forcing_realization.py` serializes that exact RNG stream
and coarse-grid normalization into a compact deterministic NPZ. The packaged
NPZ can be checked without rewriting it:

```bash
python3 generation/generate_forcing_realization.py --check
```

`generation/reference_refinement.py --dry-run` describes the independent
unforced convergence calculation. The full run is intentionally separate from
the reported-value audit because it executes six complete reference solves.

## Superseded work

The pre-correction Burgers campaigns, V5 Stage-B continuation V1 (an interface
keyword failure before scientific evaluation), and the V5 Stage-A addendum V1
are superseded. A later forced-reference refinement sensitivity study is not
the unforced refinement table in the manuscript and is deliberately excluded
from the packaged reported evidence.

The complete per-model tables for contextual data-only, regularized MFA, and
canonical controls remain archived in `results/two_d_burgers/postselection/`.
They are evidence-only here; `reported_metrics.json` focuses on the matched
control, selected hybrid, selection sensitivity, refinement, and reference
interpolation quantities used for the main and supplementary comparisons.
