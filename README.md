# PI-BSR

Research code and reproducibility materials for *A Physics-Informed
B-Spline Reconstruction of Flow Data* by Junoh Jung, David Lenz, Emil
Constantinescu, and Tom Peterka.

PI-BSR updates the coefficients of a fixed tensor-product B-spline model using
observations, strong-form equations, initial and boundary conditions, and
regional integral balances. The fitted object remains a standard MFA model.

## Reproducibility quick start

This is a curated release, not a complete replay of every experiment in the
article. See [reproduction coverage](reproducibility/REPRODUCTION_COVERAGE.md)
for the distinction between refitting models, recomputing diagnostics, and
checking archived results.

Run these commands from the repository root. First verify the metadata,
checkpoint records, point-set roles, archived-to-reported metric audits, and
SHA-256 inventory:

```sh
./reproducibility/scripts/verify_package.sh
```

Then build the three case-specific executables and run the focused C++ tests:

```sh
./reproducibility/scripts/build_and_test.sh all
```

To create the eight supported main-text numerical figures as PDFs and PNG
previews:

```sh
python3 reproducibility/scripts/make_figures.py all
```

This optional step requires NumPy and Matplotlib. The files are written below
`reproducibility/work/figures/`; see the
[figure guide](reproducibility/FIGURES.md) for individual commands, source
tables, and the precise reproduction boundary.

The selected-model runner is safe to inspect: without `--execute`, it only
prints the resolved command, weights, input paths, and deterministic runtime
settings.

```sh
python3 reproducibility/scripts/run_selected.py one_d
python3 reproducibility/scripts/run_selected.py burgers
python3 reproducibility/scripts/run_selected.py cavity
```

Full selected-model refits require the corresponding build and an explicit
`--execute` flag. They are substantially more expensive than the integrity and
unit checks. See [the reproducibility guide](reproducibility/README.md) before
running them. The
[manuscript crosswalk](reproducibility/MANUSCRIPT_CROSSWALK.md) identifies the
evidence released for each calculation-bearing figure and table, and the
[source-scope note](reproducibility/SOURCE_SCOPE.md) explains the curated
executable-code boundary.

## Repository layout

```text
mfa/mfa/                 1D convection--diffusion source tree
mfa_2Dburgers/mfa/       2D coupled Burgers source tree
mfa_NS/mfa/              lid-driven cavity source tree
third_party/              shared DIY and Eigen headers with license notices
reproducibility/          inputs, roles, coefficients, results, and audit tools
```

The numerical cases use separate MFA trees because their residual and balance
operators are case specific. The source trees contain only the case-specific
executables, focused scientific tests, and headers they use. Each tree retains
its `COPYING` file and cppoptlib notice; byte-identical DIY and Eigen headers
are stored once under `third_party/` with their license notices.

For the lid-driven-cavity example, the package distributes the fixed flow data
and its hash manifests, not the Firedrake solver, environment, or generation
scripts.

The reproducibility record includes the packaged observations, deterministic point
roles, selected and matched-control coefficients, machine-readable settings,
curated campaign evidence, and compact result-audit scripts. Archived
coefficients make it possible to inspect the reported models without repeating
every candidate sweep.
