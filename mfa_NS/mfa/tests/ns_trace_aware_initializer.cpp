#include <mfa/ns_trace_aware_initializer.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
    void require(bool condition, const std::string& message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    template <typename Function>
    void require_throw(Function function, const std::string& message)
    {
        bool threw = false;
        try
        {
            function();
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        require(threw, message);
    }

    void clear_environment()
    {
        unsetenv("MFA_NS_TRACE_AWARE_LS");
        unsetenv("MFA_NS_TRACE_AWARE_LS_ALPHA");
        unsetenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS");
    }

    mfa::NSTraceAwareLSConfig enabled_config()
    {
        mfa::NSTraceAwareLSConfig config;
        config.enabled = true;
        config.alpha = mfa::NS_TRACE_AWARE_ALPHA;
        config.expected_active_rows = 206458u;
        return config;
    }
}

int main()
{
    try
    {
        clear_environment();
        mfa::NSTraceAwareLSConfig config =
            mfa::ReadNSTraceAwareLSConfig();
        require(!config.enabled && config.alpha == 0.0,
                "unset trace-aware contract changed the legacy path");

        setenv("MFA_NS_TRACE_AWARE_LS", "0", 1);
        config = mfa::ReadNSTraceAwareLSConfig();
        require(!config.enabled,
                "exact zero unexpectedly enabled the trace-aware path");

        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "0.0001", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "alpha was accepted while the gate was disabled");
        unsetenv("MFA_NS_TRACE_AWARE_LS_ALPHA");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "206458", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "expected-row contract was accepted while the gate was disabled");

        clear_environment();
        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "0.0001", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "alpha was accepted while the gate was unset");

        clear_environment();
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "206458", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "expected-row contract was accepted while the gate was unset");

        clear_environment();
        setenv("MFA_NS_TRACE_AWARE_LS", "true", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "nonstrict trace-aware gate was accepted");

        setenv("MFA_NS_TRACE_AWARE_LS", "1", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "missing ridge coefficient was accepted");

        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "nan", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "non-finite ridge coefficient was accepted");
        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "0.0002", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "non-frozen ridge coefficient was accepted");
        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "0.0001 ", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "ridge coefficient with trailing text/space was accepted");
        setenv("MFA_NS_TRACE_AWARE_LS_ALPHA", "1e-4", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "missing expected-row contract was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "0206458", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "noncanonical expected-row contract was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "0", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "zero expected-row contract was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "+206458", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "signed expected-row contract was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "-206458", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "negative expected-row contract was accepted");
        setenv(
            "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS",
            "999999999999999999999999999999999999999999999999999999999999",
            1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "overflowing expected-row contract was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "206458 ", 1);
        require_throw(
            []() { mfa::ReadNSTraceAwareLSConfig(); },
            "expected-row contract with trailing text was accepted");
        setenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS", "206458", 1);
        config = mfa::ReadNSTraceAwareLSConfig();
        require(config.enabled && config.alpha == 1.0e-4 &&
                    config.expected_active_rows == 206458u,
                "valid trace-aware contract was rejected");

        mfa::ValidateNSTraceAwareLSContext(
            config,
            config.expected_active_rows,
            mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD,
            false,
            false,
            false);
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareLSContext(
                    config,
                    config.expected_active_rows - 1,
                    mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD,
                    false, false, false);
            },
            "wrong trace-aware active-row count was accepted");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareLSContext(
                    config,
                    config.expected_active_rows,
                    mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD - 1,
                    false, false, false);
            },
            "wrong trace-aware coefficient count was accepted");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareLSContext(
                    config,
                    config.expected_active_rows,
                    mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD,
                    true, false, false);
            },
            "checkpoint bypass was combined with Stage-A ridge");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareLSContext(
                    config,
                    config.expected_active_rows,
                    mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD,
                    false, true, false);
            },
            "baseline mode was combined with Stage-A ridge");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareLSContext(
                    config,
                    config.expected_active_rows,
                    mfa::NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD,
                    false, false, true);
            },
            "pressure-data mode was combined with trace-aware initialization");

        Eigen::SparseMatrix<double> normal(2, 2);
        normal.insert(0, 0) = 2.0;
        normal.insert(0, 1) = 0.5;
        normal.insert(1, 0) = 0.5;
        normal.insert(1, 1) = 3.0;
        normal.makeCompressed();
        mfa::ApplyNSTraceAwareRidge(normal, config);
        require(normal.coeff(0, 0) == 2.0 + 1.0e-4 &&
                    normal.coeff(1, 1) == 3.0 + 1.0e-4 &&
                    normal.coeff(0, 1) == 0.5 &&
                    normal.coeff(1, 0) == 0.5,
                "ridge did not modify only the normal-matrix diagonal");

        Eigen::SparseMatrix<double> legacy(2, 2);
        legacy.insert(0, 0) = 2.0;
        legacy.insert(1, 1) = 3.0;
        legacy.makeCompressed();
        mfa::NSTraceAwareLSConfig disabled;
        mfa::ApplyNSTraceAwareRidge(legacy, disabled);
        require(legacy.coeff(0, 0) == 2.0 && legacy.coeff(1, 1) == 3.0,
                "disabled trace-aware path modified the legacy matrix");

        Eigen::VectorXd coefficients(3);
        coefficients << -100.0, 0.25, 99.0;
        const double maximum = mfa::ValidateNSTraceAwareFieldSolve(
            "u", Eigen::Success, 5.0e-13, 17, coefficients);
        require(maximum == 100.0,
                "valid bounded solve audit reported the wrong maximum");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareFieldSolve(
                    "u", Eigen::NoConvergence, 5.0e-13, 17, coefficients);
            },
            "failed field solve was accepted");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareFieldSolve(
                    "u", Eigen::Success, 1.1e-12, 17, coefficients);
            },
            "residual outside the frozen tolerance was accepted");
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareFieldSolve(
                    "u", Eigen::Success,
                    std::numeric_limits<double>::quiet_NaN(),
                    17, coefficients);
            },
            "non-finite residual was accepted");
        coefficients(1) = 100.000001;
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareFieldSolve(
                    "v", Eigen::Success, 5.0e-13, 17, coefficients);
            },
            "out-of-box coefficient was accepted");
        coefficients(1) = std::numeric_limits<double>::infinity();
        require_throw(
            [&]() {
                mfa::ValidateNSTraceAwareFieldSolve(
                    "v", Eigen::Success, 5.0e-13, 17, coefficients);
            },
            "non-finite coefficient was accepted");

        Eigen::VectorXd pressure = Eigen::VectorXd::Zero(3);
        mfa::RequireNSTraceAwarePositiveZeroPressure(pressure);
        pressure(1) = -0.0;
        require_throw(
            [&]() {
                mfa::RequireNSTraceAwarePositiveZeroPressure(pressure);
            },
            "negative-zero pressure coefficient was accepted");

        clear_environment();
        std::cout << "NS trace-aware initializer tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        clear_environment();
        std::cerr << "NS trace-aware initializer test failed: "
                  << error.what() << "\n";
        return 1;
    }
}
