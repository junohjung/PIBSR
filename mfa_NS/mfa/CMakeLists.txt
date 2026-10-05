cmake_minimum_required(VERSION 3.15)

project(pibsr_cavity_reproducibility LANGUAGES CXX)

option(mfa_build_examples "Build the cavity driver" ON)
option(mfa_build_tests "Build the cavity verification tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

add_library(pibsr_cavity_dependencies INTERFACE)
target_compile_definitions(
    pibsr_cavity_dependencies
    INTERFACE
        MFA_SERIAL
        MFA_NO_WEIGHTS
        DIY_HAS_MPI=1
)
target_include_directories(
    pibsr_cavity_dependencies
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cppoptlib/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/diy/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/eigen3"
)
target_link_libraries(
    pibsr_cavity_dependencies
    INTERFACE
        MPI::MPI_CXX
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

if(mfa_build_examples)
    add_executable(flowdata examples/pde/flowdata.cpp)
    target_link_libraries(flowdata PRIVATE pibsr_cavity_dependencies)
    set_target_properties(
        flowdata
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/pde"
    )
endif()

if(mfa_build_tests)
    enable_testing()

    function(add_cavity_test target source test_name)
        add_executable(${target} "tests/${source}")
        target_link_libraries(${target} PRIVATE pibsr_cavity_dependencies)
        add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target}> ${ARGN})
    endfunction()

    add_cavity_test(
        ns-integral-balance-test
        test_ns_integral_balance.cpp
        ns-integral-balance-test
    )
    add_cavity_test(
        ns-knot-span-integral-balance-test
        test_ns_knot_span_integral_balance.cpp
        ns-knot-span-integral-balance-test
    )
    add_cavity_test(
        ns-knot-span-mfa-compatibility-test
        test_ns_knot_span_mfa_compatibility.cpp
        ns-knot-span-mfa-compatibility-test
    )
    add_cavity_test(
        ns-knot-span-postfit-gl8
        ns_knot_span_postfit_gl8.cpp
        ns-knot-span-postfit-gl8-self-test
        --self-test
    )
    add_cavity_test(
        ns-initializer-checkpoint-test
        ns_initializer_checkpoint.cpp
        ns-initializer-checkpoint-fail-closed
    )
    add_cavity_test(
        ns-trace-aware-initializer-test
        ns_trace_aware_initializer.cpp
        ns-trace-aware-initializer
    )
    add_cavity_test(
        ns-boundary-path-policy-test
        ns_boundary_path_policy.cpp
        ns-boundary-path-policy
    )
    add_cavity_test(
        lbfgsb-update-count-test
        lbfgsb_update_count.cpp
        lbfgsb-update-count
    )
    add_cavity_test(
        ns-hard-trace-bounds-test
        ns_hard_trace_bounds.cpp
        ns-hard-trace-bounds
    )
    add_cavity_test(
        reg-mfa-tikhonov-test
        reg_mfa_tikhonov.cpp
        reg-mfa-tikhonov
    )
    add_cavity_test(
        ns-constrained-minimax-test
        ns_constrained_minimax.cpp
        ns-constrained-minimax-v2
    )
    add_cavity_test(
        ns-constrained-adapter-test
        ns_constrained_ns_adapter.cpp
        ns-constrained-adapter-v2
    )
    add_cavity_test(
        ns-constrained-fit-evaluator-test
        ns_constrained_fit_evaluator.cpp
        ns-constrained-fit-evaluator-v2
    )
endif()
