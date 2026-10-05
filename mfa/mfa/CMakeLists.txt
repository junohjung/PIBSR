cmake_minimum_required(VERSION 3.15)
project(pibsr_convection_diffusion LANGUAGES CXX)

include(CTest)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

add_library(pibsr_mfa_headers INTERFACE)
target_compile_features(pibsr_mfa_headers INTERFACE cxx_std_11)
target_compile_definitions(
    pibsr_mfa_headers
    INTERFACE
        DIY_HAS_MPI=1
        MFA_NO_WEIGHTS
        MFA_SERIAL
)
target_include_directories(
    pibsr_mfa_headers
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cppoptlib/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/diy/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/eigen3"
)
target_link_libraries(
    pibsr_mfa_headers
    INTERFACE
        MPI::MPI_CXX
        Threads::Threads
        ${CMAKE_DL_LIBS}
)

add_executable(flowdata examples/pde/flowdata.cpp)
target_link_libraries(flowdata PRIVATE pibsr_mfa_headers)
set_target_properties(
    flowdata
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/pde"
)

add_executable(
    scalar-integral-balance-test
    tests/test_scalar_integral_balance.cpp
)
target_link_libraries(
    scalar-integral-balance-test
    PRIVATE pibsr_mfa_headers
)
add_test(
    NAME scalar-integral-balance-test
    COMMAND scalar-integral-balance-test
)

add_executable(
    weighted-least-squares-certificate-test
    tests/test_weighted_least_squares_certificate.cpp
)
target_link_libraries(
    weighted-least-squares-certificate-test
    PRIVATE pibsr_mfa_headers
)
add_test(
    NAME weighted-least-squares-certificate-test
    COMMAND weighted-least-squares-certificate-test
)
