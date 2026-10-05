#ifndef MFA_NS_TRACE_AWARE_INITIALIZER_HPP
#define MFA_NS_TRACE_AWARE_INITIALIZER_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfa
{
    // The v5 initializer requires its role size to be supplied explicitly and
    // checked against the observed mask.  The campaign runner pins 206458 for
    // CAVITY-RAMPED-GL4-v1-A2; no source-level TGV row count is inherited.
    static const std::size_t NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD = 20000u;
    static const double NS_TRACE_AWARE_ALPHA = 1.0e-4;
    static const double NS_TRACE_AWARE_COEFFICIENT_BOUND = 100.0;
    static const double NS_TRACE_AWARE_SOLVER_RESIDUAL_BOUND = 1.0e-12;

    struct NSTraceAwareLSConfig
    {
        bool enabled;
        double alpha;
        std::size_t expected_active_rows;

        NSTraceAwareLSConfig()
            : enabled(false), alpha(0.0), expected_active_rows(0u) {}
    };

    inline std::size_t ParseNSTraceAwareExpectedRowsStrict(const char* text)
    {
        if (text == NULL || *text == '\0')
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS is required when "
                "MFA_NS_TRACE_AWARE_LS=1");
        const std::string token(text);
        if (token.empty() || (token.size() > 1u && token[0] == '0'))
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS must be a canonical "
                "positive integer");
        std::size_t value = 0u;
        for (std::size_t i = 0; i < token.size(); ++i)
        {
            const char digit = token[i];
            if (digit < '0' || digit > '9')
                throw std::invalid_argument(
                    "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS must be a "
                    "canonical positive integer");
            const std::size_t component =
                static_cast<std::size_t>(digit - '0');
            if (value > (std::numeric_limits<std::size_t>::max() - component) /
                            static_cast<std::size_t>(10u))
                throw std::invalid_argument(
                    "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS is out of range");
            value = value * static_cast<std::size_t>(10u) + component;
        }
        if (value == 0u)
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS must be positive");
        return value;
    }

    inline double ParseNSTraceAwareAlphaStrict(const char* text)
    {
        if (text == NULL || *text == '\0')
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS_ALPHA is required when "
                "MFA_NS_TRACE_AWARE_LS=1");
        errno = 0;
        char* end = NULL;
        const double value = std::strtod(text, &end);
        if (errno == ERANGE || end == text || *end != '\0' ||
            !std::isfinite(value) || !(value > 0.0))
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS_ALPHA must be one finite positive "
                "numeric token");
        if (value != NS_TRACE_AWARE_ALPHA)
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS_ALPHA must equal the frozen "
                "ramped-cavity value 0.0001");
        return value;
    }

    inline NSTraceAwareLSConfig ReadNSTraceAwareLSConfig()
    {
        NSTraceAwareLSConfig config;
        const char* gate = std::getenv("MFA_NS_TRACE_AWARE_LS");
        const char* alpha = std::getenv("MFA_NS_TRACE_AWARE_LS_ALPHA");
        const char* expected_rows =
            std::getenv("MFA_NS_TRACE_AWARE_EXPECTED_ACTIVE_ROWS");

        if (gate == NULL)
        {
            if (alpha != NULL || expected_rows != NULL)
                throw std::invalid_argument(
                    "A trace-aware initializer parameter is defined while "
                    "MFA_NS_TRACE_AWARE_LS is unset");
            return config;
        }
        if (std::string(gate) == "0")
        {
            if (alpha != NULL || expected_rows != NULL)
                throw std::invalid_argument(
                    "A trace-aware initializer parameter is defined while "
                    "MFA_NS_TRACE_AWARE_LS=0");
            return config;
        }
        if (std::string(gate) != "1")
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS must be exactly 0 or 1");

        config.enabled = true;
        config.alpha = ParseNSTraceAwareAlphaStrict(alpha);
        config.expected_active_rows =
            ParseNSTraceAwareExpectedRowsStrict(expected_rows);
        return config;
    }

    inline void ValidateNSTraceAwareLSContext(
        const NSTraceAwareLSConfig& config,
        std::size_t active_rows,
        std::size_t coefficients_per_field,
        bool checkpoint_bypass,
        bool baseline_requested,
        bool pressure_uses_data)
    {
        if (!config.enabled)
            return;
        if (checkpoint_bypass)
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS cannot be combined with "
                "MFA_INIT_CHECKPOINT_IN; checkpoint preload must bypass the "
                "Stage-A ridge solve");
        if (baseline_requested)
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS cannot be combined with corrected "
                "Reg-MFA baseline mode");
        if (pressure_uses_data)
            throw std::invalid_argument(
                "MFA_NS_TRACE_AWARE_LS requires withheld pressure and an "
                "exact positive-zero pressure initializer");
        if (config.expected_active_rows == 0u)
            throw std::runtime_error(
                "Trace-aware NS initializer lacks an expected active-row "
                "contract");
        if (active_rows != config.expected_active_rows)
            throw std::runtime_error(
                "Trace-aware NS initializer active-row mismatch: observed " +
                std::to_string(active_rows) + ", expected " +
                std::to_string(config.expected_active_rows));
        if (coefficients_per_field !=
            NS_TRACE_AWARE_COEFFICIENTS_PER_FIELD)
            throw std::runtime_error(
                "Trace-aware NS initializer requires exactly 20000 "
                "coefficients per field");
    }

    template <typename Scalar, int Options, typename StorageIndex>
    inline void ApplyNSTraceAwareRidge(
        Eigen::SparseMatrix<Scalar, Options, StorageIndex>& matrix,
        const NSTraceAwareLSConfig& config)
    {
        if (!config.enabled)
            return;
        if (matrix.rows() != matrix.cols() || matrix.rows() == 0)
            throw std::runtime_error(
                "Trace-aware NS ridge requires a nonempty square normal "
                "matrix");
        for (Eigen::Index i = 0; i < matrix.rows(); ++i)
            matrix.coeffRef(i, i) += static_cast<Scalar>(config.alpha);
        matrix.makeCompressed();
    }

    template <typename Derived>
    inline double NSTraceAwareCoefficientMaxAbs(
        const Eigen::MatrixBase<Derived>& coefficients)
    {
        if (coefficients.size() == 0)
            throw std::runtime_error(
                "Trace-aware NS initializer produced an empty coefficient "
                "field");
        double maximum = 0.0;
        for (Eigen::Index i = 0; i < coefficients.size(); ++i)
        {
            const double value = static_cast<double>(coefficients.derived()(i));
            if (!std::isfinite(value))
                throw std::runtime_error(
                    "Trace-aware NS initializer produced a non-finite "
                    "coefficient");
            maximum = std::max(maximum, std::abs(value));
        }
        return maximum;
    }

    template <typename Derived>
    inline double ValidateNSTraceAwareFieldSolve(
        const char* field,
        Eigen::ComputationInfo status,
        double residual,
        int iterations,
        const Eigen::MatrixBase<Derived>& coefficients)
    {
        const std::string label = field == NULL ? "unknown" : field;
        if (status != Eigen::Success)
            throw std::runtime_error(
                "Trace-aware NS initializer " + label +
                " solve did not converge");
        if (!std::isfinite(residual) || residual < 0.0 ||
            residual > NS_TRACE_AWARE_SOLVER_RESIDUAL_BOUND)
            throw std::runtime_error(
                "Trace-aware NS initializer " + label +
                " solve residual is outside the frozen [0,1e-12] gate");
        if (iterations < 0)
            throw std::runtime_error(
                "Trace-aware NS initializer " + label +
                " solve reported a negative iteration count");
        const double maximum =
            NSTraceAwareCoefficientMaxAbs(coefficients);
        if (maximum > NS_TRACE_AWARE_COEFFICIENT_BOUND)
            throw std::runtime_error(
                "Trace-aware NS initializer " + label +
                " coefficient violates the frozen [-100,100] box");
        return maximum;
    }

    template <typename Derived>
    inline void RequireNSTraceAwarePositiveZeroPressure(
        const Eigen::MatrixBase<Derived>& coefficients)
    {
        if (coefficients.size() == 0)
            throw std::runtime_error(
                "Trace-aware NS initializer pressure field is empty");
        for (Eigen::Index i = 0; i < coefficients.size(); ++i)
        {
            const double value = static_cast<double>(coefficients.derived()(i));
            if (value != 0.0 || std::signbit(value))
                throw std::runtime_error(
                    "Trace-aware NS initializer pressure coefficients must "
                    "be exact positive zero");
        }
    }
}

#endif
