#ifndef MFA_NS_INITIALIZER_CHECKPOINT_HPP
#define MFA_NS_INITIALIZER_CHECKPOINT_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa
{
    struct NSInitializerCheckpointPlan
    {
        bool bypass_data_solve;
        std::string path;

        NSInitializerCheckpointPlan() : bypass_data_solve(false) {}
    };

    // MFA_INIT_CHECKPOINT_IN is itself the explicit request to bypass the
    // legacy data-only normal-equation solve.  A defined-but-empty value is an
    // error: silently falling back to the solve would defeat the fail-closed
    // Stage-B checkpoint contract.
    inline NSInitializerCheckpointPlan ReadNSInitializerCheckpointPlan()
    {
        NSInitializerCheckpointPlan plan;
        const char* value = std::getenv("MFA_INIT_CHECKPOINT_IN");
        if (value == NULL)
            return plan;
        if (*value == '\0')
            throw std::invalid_argument(
                "MFA_INIT_CHECKPOINT_IN is defined but empty; refusing "
                "data-only-solve fallback");
        plan.bypass_data_solve = true;
        plan.path = value;
        return plan;
    }

    inline std::vector<double> ReadNSInitializerCheckpointV1Strict(
        const std::string& path,
        const std::string& expected_pressure_regime,
        std::size_t expected_coefficient_count,
        double coefficient_abs_bound)
    {
        if (path.empty())
            throw std::invalid_argument(
                "NS initializer checkpoint path must not be empty");
        if (expected_pressure_regime != "zero" &&
            expected_pressure_regime != "data")
            throw std::invalid_argument(
                "NS initializer checkpoint pressure regime must be zero or data");
        if (expected_coefficient_count == 0)
            throw std::invalid_argument(
                "NS initializer checkpoint coefficient count must be positive");
        if (expected_coefficient_count % 3 != 0)
            throw std::invalid_argument(
                "NS initializer checkpoint coefficient count must contain "
                "equal u, v, and p fields");
        if (!std::isfinite(coefficient_abs_bound) ||
            !(coefficient_abs_bound > 0.0))
            throw std::invalid_argument(
                "NS initializer checkpoint coefficient bound must be finite "
                "and positive");

        std::ifstream stream(path.c_str());
        if (!stream)
            throw std::runtime_error(
                std::string("Cannot open MFA_INIT_CHECKPOINT_IN: ") + path);

        std::ostringstream expected_header;
        expected_header
            << "# PIMFA_NS_INIT_V1 pressure=" << expected_pressure_regime
            << " n=" << expected_coefficient_count;
        std::string header;
        if (!std::getline(stream, header) || header != expected_header.str())
            throw std::runtime_error(
                std::string(
                    "Initializer checkpoint header mismatch (expected exact '") +
                expected_header.str() + "'): " + path);

        std::vector<double> coefficients(expected_coefficient_count);
        const std::size_t pressure_begin =
            2 * (expected_coefficient_count / 3);
        for (std::size_t i = 0; i < coefficients.size(); ++i)
        {
            if (!(stream >> coefficients[i]) ||
                !std::isfinite(coefficients[i]))
                throw std::runtime_error(
                    std::string(
                        "Initializer checkpoint has a truncated or non-finite "
                        "payload: ") + path);
            if (std::abs(coefficients[i]) > coefficient_abs_bound)
                throw std::runtime_error(
                    std::string(
                        "Initializer checkpoint coefficient violates the "
                        "L-BFGS-B feasible bound: ") + path);
            if (expected_pressure_regime == "zero" &&
                i >= pressure_begin &&
                (coefficients[i] != 0.0 || std::signbit(coefficients[i])))
                throw std::runtime_error(
                    std::string(
                        "Initializer checkpoint zero-pressure payload must "
                        "contain exact +0 coefficients: ") + path);
        }

        std::string extra;
        if (stream >> extra)
            throw std::runtime_error(
                std::string(
                    "Initializer checkpoint contains an extra payload token: ") +
                path);
        return coefficients;
    }
}

#endif
