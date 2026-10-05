#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${script_dir}/.." && pwd)"

cd "${package_dir}"
python3 validate_package.py --manifest SHA256SUMS
python3 calculations/one_d_convection_diffusion/audit_results.py
python3 calculations/two_d_burgers/audit_results.py
python3 calculations/lid_driven_cavity/audit_results.py
python3 calculations/verification/audit_results.py
python3 -m unittest discover -s tests -v
