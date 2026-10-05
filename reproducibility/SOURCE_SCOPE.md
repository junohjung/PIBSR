# Source-code scope

This is a target-focused research release, not a copy of the full upstream MFA
repository. Source selection was traced from clean Release builds of the three
case-specific executables and their focused scientific tests.

| Case source | Case executable | Focused tests |
|---|---|---:|
| `mfa/mfa/` | 1D convection--diffusion `flowdata` | 2 |
| `mfa_2Dburgers/mfa/` | two-dimensional Burgers `flowdata` | 3 |
| `mfa_NS/mfa/` | lid-driven-cavity `flowdata` | 13 |

Each source tree contains a compact CMake project that declares only those
targets. The case drivers and data readers accept only `ConDiff1D`, `Burger2D`,
or `NS2D`, respectively. Their numerical optimization kernels and environment
variable contracts are retained from the calculation branches used for the
revision. Input readers now reject missing, truncated, nonnumeric, nonfinite,
or overlong data instead of continuing with a partial parse. Repeated reads of
time-independent boundary files were replaced by one checked read followed by
time-indexed replication.

General MFA examples, converters, bindings, packaging files, benchmarks,
unselected tests, superseded source copies, and build artifacts are excluded.
Case-local MFA and optimizer headers remain where they are compilation
dependencies. The byte-identical DIY and Eigen snapshots are stored once in
`third_party/`; their full header subtrees are retained for platform-dependent
compilation, together with their license notices. The separate external `fmt`
build is not required by these header-only release targets and is omitted.

The Firedrake program and environment that generated the cavity observations
are also outside this release. Only the fixed cavity flow data and provenance
manifests are distributed. This boundary does not affect the PI-BSR fitting
executable, which consumes those packaged data files.

Run `./reproducibility/scripts/build_and_test.sh all` from the repository root
to configure clean case builds and execute all 18 focused C++ tests. Package
metadata, data hashes, archived-to-reported calculations, and Python validator
tests are checked separately by `./reproducibility/scripts/verify_package.sh`.
