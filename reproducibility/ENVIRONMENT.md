# Environment record

The package separates lightweight verification from numerical replay. No
third-party Python package is needed to verify metadata and hashes. Building or
rerunning a model requires a C++/MPI toolchain, and a few optional calculation
scripts require NumPy.

## Requirements by task

| Task | Requirements |
|---|---|
| `scripts/verify_package.sh` | Bash and Python 3.9 or newer; Python standard library only |
| `scripts/build_and_test.sh` | Bash, CMake, a C++ compiler exposed through an MPI wrapper such as `mpicxx`, and the repository's vendored headers |
| `scripts/run_selected.py` | a completed case build plus Python 3.9 or newer |
| `scripts/make_figures.py` | Python 3.9 or newer and `requirements-figures.txt` |
| `scripts/make_figure5.py` | Python 3.12 or newer and `requirements-figure5.txt`; Poppler is optional for PNG previews |
| calculation audits | Python 3.9 or newer; scripts that read or write NPZ arrays also require NumPy |
| regenerate Burgers observations | NumPy; see `calculations/two_d_burgers/README.md` |

The packaged observations and references are sufficient for the selected-model
replays. The Firedrake/PETSc generator and environment used to produce the
cavity inputs are outside the distributed package; only the resulting data and
provenance metadata are included.

## Locally verified build environment

The release preparation used the following environment on 2026-09-02:

| Component | Verified value |
|---|---|
| Operating system | macOS 15.7.9, arm64 (`Darwin 24.6.0`) |
| CMake | 4.2.1 |
| C++ compiler | Apple clang 17.0.0 (`clang-1700.0.13.5`) |
| MPI wrapper/runtime | Open MPI 5.0.9, `mpicxx` / `mpiexec` |
| Build type | Release, `-O3 -DNDEBUG` |
| Package-validation Python | CPython 3.14.2 |

The focused targets were configured as Release builds with the repository's
shared Eigen and DIY headers and case-local MFA and cppoptlib headers. Coin-OR
CLP was not present
and is not required by the focused release targets. In this environment the
release checks passed for 2/2 focused 1D tests, 3/3 Burgers tests, and 13/13
cavity tests.

The selected-model runner removes inherited `MFA_*` settings, applies only the
recorded case contract, fixes common OpenMP/BLAS thread variables to one, and
uses the local MPI self transport. Those controls reduce accidental variation;
they do not guarantee bitwise equality across compilers or platforms.

## Archived production evidence

Archived records describe the completed scientific campaigns. They should not
be conflated with the local package-build environment above.

The 1D freeze and sparse-QR certificate record:

- macOS 15.7.9 on arm64;
- Apple clang 17.0.0 through Open MPI 5.0.9 `mpicxx`;
- a Release build with `-O3 -DNDEBUG`;
- CPython 3.12.13 and NumPy 2.3.5 for the orchestration and audit layer;
- vendored Eigen version 3.3.90.

The archived cavity post-selection command audits record CPython 3.12.8 and
NumPy 2.2.1. The Burgers records preserve the exact optimizer environment and
input hashes but not a complete independent software-version inventory.

Independent regeneration of the cavity observations is outside the package
scope. The supplied observations are hash sealed so integrity checks and
selected-model replays use the same fixed inputs as the revision.

Hardware timing and peak-memory values are machine dependent. The package does
not present the local verification machine as a replacement for the production
hardware record, and rebuilding the sources alone is not evidence of bitwise
reproduction of a completed optimization.
