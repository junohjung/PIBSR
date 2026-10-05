# Lid-driven-cavity flow data

This directory contains the fixed flow inputs used by the PI-BSR cavity
example. It intentionally contains data only; the Firedrake solver,
environment, and generation scripts are not distributed in this reproducibility
package.

- `snapshots/` contains 1,001 endpoint-inclusive coarse velocity snapshots.
- `dynamic_boundaries/` contains the time-indexed x- and y-velocity boundary
  records consumed by the fitting executable.
- `reporting_plan.npz` contains deterministic coordinates and index roles for
  post-selection reporting, not an additional flow solution.
- `selection_input_manifest_a2.json`, `dynamic_boundaries_manifest.json`, and
  `reporting_plan_manifest.json` record input contracts and SHA-256 digests.

The configuration and historical generator/protocol digests retained in the
manifests document provenance. They do not make the omitted Firedrake code an
executable dependency. From the repository root, run
`./reproducibility/scripts/verify_package.sh` to verify all packaged files.
