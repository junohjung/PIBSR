#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${package_dir}/.." && pwd)"
build_root="${PIMFA_BUILD_ROOT:-${repo_dir}/.build}"
jobs="${PIMFA_BUILD_JOBS:-2}"

usage() {
    echo "usage: $0 [one_d|burgers|cavity|all]" >&2
}

compiler="${PIMFA_CXX:-}"
if [[ -z "${compiler}" ]]; then
    compiler="$(command -v mpicxx || true)"
fi
if [[ -z "${compiler}" ]]; then
    echo "MPI C++ compiler not found; set PIMFA_CXX to its path." >&2
    exit 1
fi

build_case() {
    local case_name="$1"
    local source_dir
    local build_dir
    local test_pattern
    local targets=()

    case "${case_name}" in
        one_d)
            source_dir="${repo_dir}/mfa/mfa"
            build_dir="${build_root}/one_d"
            targets=(
                flowdata
                scalar-integral-balance-test
                weighted-least-squares-certificate-test
            )
            test_pattern='^(scalar-integral-balance-test|weighted-least-squares-certificate-test)$'
            ;;
        burgers)
            source_dir="${repo_dir}/mfa_2Dburgers/mfa"
            build_dir="${build_root}/burgers"
            targets=(
                flowdata
                reg-mfa-tikhonov-test
                burgers-knot-span-integral-balance-test
                burgers-boundary-contract-test
            )
            test_pattern='^(reg-mfa-tikhonov|burgers-knot-span-integral-balance|burgers-boundary-contract)$'
            ;;
        cavity)
            source_dir="${repo_dir}/mfa_NS/mfa"
            build_dir="${build_root}/cavity"
            targets=(
                flowdata
                ns-integral-balance-test
                ns-knot-span-integral-balance-test
                ns-knot-span-mfa-compatibility-test
                ns-knot-span-postfit-gl8
                ns-initializer-checkpoint-test
                ns-trace-aware-initializer-test
                ns-boundary-path-policy-test
                lbfgsb-update-count-test
                ns-hard-trace-bounds-test
                reg-mfa-tikhonov-test
                ns-constrained-minimax-test
                ns-constrained-adapter-test
                ns-constrained-fit-evaluator-test
            )
            test_pattern='^(ns-integral-balance-test|ns-knot-span-integral-balance-test|ns-knot-span-mfa-compatibility-test|ns-knot-span-postfit-gl8-self-test|ns-initializer-checkpoint-fail-closed|ns-trace-aware-initializer|ns-boundary-path-policy|lbfgsb-update-count|ns-hard-trace-bounds|reg-mfa-tikhonov|ns-constrained-minimax-v2|ns-constrained-adapter-v2|ns-constrained-fit-evaluator-v2)$'
            ;;
        *)
            usage
            exit 2
            ;;
    esac

    echo "Configuring ${case_name} in ${build_dir}"
    cmake \
        -S "${source_dir}" \
        -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${compiler}"

    echo "Building ${case_name} targets"
    cmake --build "${build_dir}" --parallel "${jobs}" --target "${targets[@]}"

    echo "Testing ${case_name}"
    ctest --test-dir "${build_dir}" -R "${test_pattern}" --output-on-failure
}

requested="${1:-all}"
if [[ "$#" -gt 1 ]]; then
    usage
    exit 2
fi

if [[ "${requested}" == "all" ]]; then
    build_case one_d
    build_case burgers
    build_case cavity
else
    build_case "${requested}"
fi
