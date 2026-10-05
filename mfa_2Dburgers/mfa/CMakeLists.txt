cmake_minimum_required(VERSION 3.15)

project(pibsr_burgers_reproducibility LANGUAGES CXX)

option(mfa_build_examples "Build the selected Burgers solver" ON)
option(mfa_build_tests "Build the selected Burgers tests" ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

add_library(pibsr_burgers_headers INTERFACE)
target_compile_features(pibsr_burgers_headers INTERFACE cxx_std_11)
target_compile_definitions(
    pibsr_burgers_headers
    INTERFACE
        MFA_NO_WEIGHTS
        MFA_SERIAL
        DIY_HAS_MPI=1
)
target_include_directories(
    pibsr_burgers_headers
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cppoptlib/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/diy/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/eigen3
        ${CMAKE_CURRENT_SOURCE_DIR}/examples
)
target_link_libraries(
    pibsr_burgers_headers
    INTERFACE
        MPI::MPI_CXX
        Threads::Threads
)

if(mfa_build_examples)
    add_executable(flowdata examples/pde/flowdata.cpp)
    target_link_libraries(flowdata PRIVATE pibsr_burgers_headers)
    set_target_properties(
        flowdata
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/pde"
    )
endif()

if(mfa_build_tests)
    enable_testing()

    add_executable(reg-mfa-tikhonov-test tests/reg_mfa_tikhonov.cpp)
    target_link_libraries(reg-mfa-tikhonov-test PRIVATE pibsr_burgers_headers)

    add_executable(
        burgers-knot-span-integral-balance-test
        tests/test_burgers_knot_span_integral_balance.cpp
    )
    target_link_libraries(
        burgers-knot-span-integral-balance-test
        PRIVATE pibsr_burgers_headers
    )

    add_executable(
        burgers-boundary-contract-test
        tests/test_burgers_boundary_contract.cpp
    )
    target_link_libraries(
        burgers-boundary-contract-test
        PRIVATE pibsr_burgers_headers
    )

    add_test(NAME reg-mfa-tikhonov COMMAND reg-mfa-tikhonov-test)
    add_test(
        NAME burgers-knot-span-integral-balance
        COMMAND burgers-knot-span-integral-balance-test
    )
    add_test(
        NAME burgers-boundary-contract
        COMMAND burgers-boundary-contract-test
    )
endif()
