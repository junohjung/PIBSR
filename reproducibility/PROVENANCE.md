# Provenance and curation

This release was assembled from the final revision campaigns used by the
current manuscript and supplementary material. Superseded trials were compared
against the final manuscript settings and excluded from the executable path;
they are retained only where they are useful for explicit lineage or
sensitivity records.

## Source lineage

The released case trees come from these final numerical branches of the
revision work:

- 1D convection--diffusion: the final v14 integral-balance source;
- 2D Burgers: the v14 boundary-corrected Burgers source;
- lid-driven cavity: the final smooth-ramp GL4 cavity source, with the later v14
  post-selection calculation records used by the revised figures and tables.

The case README files under `calculations/` give the exact archived script and
result lineage, including rejected or superseded campaigns where that
distinction matters. Each source tree retains its upstream MFA `COPYING` file
and the notice for its case-local cppoptlib subset. The shared DIY and Eigen
license notices are under `third_party/`.

The lid-driven-cavity observations are distributed as fixed, hash-sealed data.
The Firedrake solver, its environment, and its input-generation source are not
part of this reproducibility package. Historical source and protocol digests remain
in the data manifest solely to identify the omitted provenance record.

References of the form `PIBSR_*_v2.tex:line` in configurations and result
metadata are stable identifiers for the submitted manuscript source; they are
not repository-relative paths and the TeX submission files are not duplicated
in this code package.

## Executable record versus archived evidence

Portable execution is defined by:

- `configs/*.json` for the scientific setup;
- `point_sets/` for fitting and validation roles;
- `checkpoints/manifest.json` and the referenced coefficient files;
- `scripts/run_selected.py` for repository-relative commands and environment
  overrides.

Files under `results/` preserve numerical records from completed runs.
Personal absolute paths in historical metadata are replaced by inert
`historical_source/` or `historical_temporary/` labels. These labels identify
source lineage, not executable paths or additional packaged files.
Non-scientific annotations are omitted. Numerical values and scientific settings are
unchanged. Nested historical digests identify the original upstream records;
the release checksums identify the distributed, normalized copies.

The two byte-identical copies of the 1D selection configuration omit the
non-scientific `runtime.python_executable` absolute path. Their remaining
runtime version fields and every scientific field are unchanged. Historical
`freeze_sha256` values in downstream records continue to identify the original
upstream freeze (`c58bc4092598a66d892f02e574892b5a4de1efbbdf5bb8c8db3573f424a7c02f`);
the package-level `SHA256SUMS` identifies the distributed copies.

## Curated boundary

The result archive includes the records needed to establish:

- which candidate families and controls were compared;
- which fit and validation roles were held fixed;
- which candidate was selected without reference data;
- the coefficients and loss histories for selected or matched runs;
- the post-selection field, residual, balance, pressure, and independent-check
  values reported in the revision.

The package omits build directories, temporary solver output, redundant plots,
and superseded campaign copies that do not contribute to lineage. This is
therefore a curated evidence set, not a mirror of the entire working revision
directory. Original campaign manifests are retained where available, while
`SHA256SUMS` covers every released regular file below `reproducibility/`. The
narrower `checkpoints/manifest.json` additionally binds each coefficient file
to its case and scientific role.

## Verification scope

Package assembly included metadata validation, checksum verification, selected
command dry runs, Release builds, and focused C++ tests. No full expensive
selected-model refit and no complete hyperparameter sweep was rerun during
assembly. The archived coefficients and result records are provided precisely
so users can inspect or postprocess the reported models before deciding
whether to repeat those computations.

The higher-resolution cavity trajectory is a numerical comparison rather than
a formal convergence certificate. The Burgers unforced reference and the 1D
analytical reference are post-selection evidence and were not used to choose
weights. Contextual PINN results are likewise separated from PI-BSR model
selection.
