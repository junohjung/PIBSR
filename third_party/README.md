# Shared header dependencies

The three case-specific source trees use the same snapshots of DIY and Eigen.
They are stored once here to avoid shipping three byte-identical copies.

- `diy/` contains the header subtree used by the MPI block orchestration code
  and its BSD-style license.
- `eigen3/` contains the Eigen 3.3.90 header subtree and the license notices
  supplied with that snapshot.

Examples, tests, documentation, package metadata, and build machinery from
these dependencies are omitted because the reviewer targets do not use them.
