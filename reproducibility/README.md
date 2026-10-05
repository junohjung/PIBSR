# PI-BSR reproducibility package

This directory contains the numerical record used for the revised PI-BSR
manuscript and supplementary material. It is organized so that fitting,
model-selection validation, and post-selection reporting remain separate.

## Package map

- `configs/` records the equations, grids, spline spaces, weights, optimizer
  settings, role counts, and reported evaluation designs for each case.
- `data/` contains the packaged observations and post-selection references.
- `point_sets/` contains the deterministic fitting and validation indices.
- `checkpoints/` contains Stage-A initializers and the selected or matched
  coefficient vectors; `checkpoints/manifest.json` records their roles, sizes,
  and hashes.
- `results/` contains a curated subset of the completed revision campaigns.
- `figure_data/` contains compact tables used directly by the figure
  renderer.
- `calculations/` contains compact, readable scripts that audit the reported
  case and package-wide verification numbers, with README files that document
  their archived calculation lineage.
- `expected_results.json` is the package-wide acceptance record used by the
  validator. The `calculations/*/reported_metrics.json` files are the
  corresponding case-level and verification extracts.
- `SHA256SUMS` covers every released regular file below this directory except
  itself. Generated `work/`, `__pycache__/`, and `.pyc` files are excluded.

[MANUSCRIPT_CROSSWALK.md](MANUSCRIPT_CROSSWALK.md) maps each calculation-bearing
figure and table to the released evidence and states where the package is only
partial or contextual. [RESULTS.md](RESULTS.md) summarizes the main
comparisons, [ENVIRONMENT.md](ENVIRONMENT.md) records software requirements and
known environments, [PROVENANCE.md](PROVENANCE.md) explains how the archived
records were selected, and [SOURCE_SCOPE.md](SOURCE_SCOPE.md) documents the
target-level source-code boundary. [FIGURES.md](FIGURES.md) gives the supported
figure commands and their exact scope. [RELEASE_SCOPE.md](RELEASE_SCOPE.md)
states the distribution and licensing boundaries of this public release.

Metadata strings such as `PIBSR_manuscript_v2.tex:891-922` identify locations
in the submitted manuscript source. They are provenance labels, not links to
files inside this code package.

## Keep the numerical roles separate

| Case | Executable fitting contract | Weight selection | Post-selection reporting |
|---|---|---|---|
| 1D convection--diffusion | full `60 x 400` trajectory with IC/BC traces; 279 balance-fit times for the hybrid stage | beta on the `59 x 399` midpoint data/PDE grid; gamma on 120 separate balance-selection times | analytical reference, 94,164 cellwise GL2 points, and 400 whole-domain GL8 times |
| 2D Burgers | 106,213 data, 106,669 PDE, 1,681 IC, and 32,964 ordered BC rows; the optimizer's integral block uses 140 fit and 60 disjoint internal-validation times | data/PDE validation and trace guards; gamma on sealed Design A with 200 off-node times | disjoint sealed Design B with 200 balance-confirmation times, plus off-node fields, mechanics, and the unforced reference |
| Lid-driven cavity | 126,017 velocity-data and 126,886 PDE rows plus IC, BC, and gauge terms; 700 balance-fit times | data/PDE validation, off-grid traces, and 300 disjoint balance-validation times | 200 sealed balance midpoints, the velocity test set, pressure comparison, shifted regions, and consistency checks |

Burgers has three distinct temporal roles: the optimizer's internal 140/60
fit/validation files, the 200-time sealed Design A used to rank gamma
candidates, and the disjoint 200-time sealed Design B used to confirm the
selected candidate. The selected-model runner passes only the 140/60 role files
to the executable. Neither 200-time design should be substituted for those
files.

For the cavity, `data/lid_driven_cavity/reporting_plan.npz` is a coordinate-only
post-selection/reference plan. It also preserves `conservative_*` arrays from
the reporting-plan construction, but those arrays are not the executable
campaign split. Reproduction of the selected fit uses only the 700/300 files in
`point_sets/lid_driven_cavity/integral_time_roles/`, as can be seen in
`scripts/run_selected.py`.

References were not used to select weights. The 1D analytical solution, the
Burgers unforced trajectory, and the higher-resolution cavity solution enter
only after model selection.

For the lid-driven-cavity example, this release contains the generated flow
data and the metadata needed to verify those fixed inputs. The Firedrake
solver, environment, and generation scripts are intentionally excluded.

## 1. Verify the package

From the repository root:

```sh
./reproducibility/scripts/verify_package.sh
```

This command uses the Python standard library to check the three
configurations, reported-value arithmetic, model identifiers, point-set counts
and disjointness, checkpoint sizes and hashes, and the complete global
`SHA256SUMS` inventory. It then audits every case-level reported metric and the
package-wide verification values against their archived source artifacts, and
runs the validator's unit tests. Each audit prints its current check count. It
does not execute an optimization.

Users should run this check before changing or regenerating any file.
`scripts/generate_checksums.py` is a maintainer utility for rebuilding the
curated subset inventories and global manifest after an intentional package
change; it is not part of initial verification.

`scripts/create_release_archive.py` is the release-preparation utility used to
create the deterministic ZIP, its archive-level SHA-256 sidecar, and the full
size/path listing. It admits only the documented repository files and source
trees, rejects symbolic links and unsafe paths, and verifies every ZIP entry's
CRC before writing the reports.

## 2. Build and run the focused tests

```sh
./reproducibility/scripts/build_and_test.sh all
```

Individual cases are `one_d`, `burgers`, and `cavity`. The script uses
`mpicxx` by default, creates Release builds under `.build/`, and builds only the
case executable and focused scientific tests. These optional variables change
the local build location, compiler, or parallelism:

```sh
PIMFA_BUILD_ROOT=/path/to/build PIMFA_CXX=/path/to/mpicxx \
  PIMFA_BUILD_JOBS=4 ./reproducibility/scripts/build_and_test.sh all
```

## 3. Inspect the selected-model contracts

Dry runs print JSON and do not create output or launch a solver:

```sh
python3 reproducibility/scripts/run_selected.py one_d
python3 reproducibility/scripts/run_selected.py burgers
python3 reproducibility/scripts/run_selected.py cavity
```

The calculation audits have their own `--help` and `--dry-run` modes:

```sh
python3 reproducibility/calculations/one_d_convection_diffusion/audit_results.py --dry-run
python3 reproducibility/calculations/two_d_burgers/audit_results.py --dry-run
python3 reproducibility/calculations/lid_driven_cavity/audit_results.py --dry-run
python3 reproducibility/calculations/verification/audit_results.py --dry-run
```

See the case README files under `calculations/` for the exact archived inputs
used by each audit and for optional data-generation or postprocessing steps.

## 4. Create the supported figures

With NumPy and Matplotlib available, create all eight supported main-text
numerical figures with:

```sh
python3 reproducibility/scripts/make_figures.py all
```

PDFs, PNG previews, and a hash manifest are written to
`reproducibility/work/figures/`. This command reads packaged observations and
figure-ready tables; it does not refit a model, rerun a candidate campaign, or
regenerate the cavity flow data. See [FIGURES.md](FIGURES.md) for case-specific
commands and the figures that remain outside the curated release.

## 5. Run a selected-model refit

Build the relevant case first, then add `--execute`:

```sh
python3 reproducibility/scripts/run_selected.py one_d --execute
python3 reproducibility/scripts/run_selected.py burgers --execute
python3 reproducibility/scripts/run_selected.py cavity --execute
```

By default, new files go to `reproducibility/work/<case>_selected/`; that tree is
excluded from package integrity checks. The runner refuses to reuse an existing
output directory. Use `--output /new/output/path` to choose a fresh location.
It fixes OpenMP and common BLAS thread counts to one and runs a single MPI
process path.

These commands refit the three selected PI-BSR models, not every candidate sweep.
No full PI-BSR refit was performed while assembling this package; source builds,
focused tests, metadata validation, and dry-run contracts verify the release structure.

## Optional contextual PINN reproduction

See [`pinn/README.md`](pinn/README.md) for the 11 original checkpoints, fixed inputs,
native evaluation and figure-curve verification. The evaluators reproduce the
reported PINN curves from the original models. Training from initialization
is outside the scope of this archive.

## Curated result scope

The `results/` tree is deliberately smaller than the revision workspace:

- the 1D subset retains the integral-balance family, selection seal,
  independent sparse-QR certificate, final post-selection metric extracts,
  and reproducibility records;
- the Burgers subset retains the final beta and Stage-A gamma selection seals,
  normalization and kappa screens, the Design-B Stage-B trajectories,
  post-selection metrics, and the unforced-reference refinement record; an
  earlier gamma family is retained only as labelled lineage and sensitivity
  evidence;
- the cavity subset retains the selected family record, matched strong and GL4
  hybrid runs, validation metrics, and raw arrays or summaries needed for the
  reported field, balance, pressure, and consistency calculations.

Build trees, temporary outputs, superseded campaigns, and redundant rendered
figures are not release artifacts. Campaign-level inventories where available,
together with the complete global `SHA256SUMS`, make the boundary of the
curated record explicit.

## Interpretation limits

- Burgers observations use disclosed deterministic forcing, whereas PI-BSR is
  evaluated against the prescribed unforced operator. This is a
  prescribed-model projection, not governing-equation discovery.
- The higher-resolution cavity solution is a numerical comparison, not ground
  truth or a formal convergence certificate.
- PINN values are contextual and do not establish a representation ranking.
