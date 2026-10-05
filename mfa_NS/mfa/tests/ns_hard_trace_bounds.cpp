#include <mfa/ns_hard_trace_bounds.hpp>

#include <LBFGSB.h>
#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <limits>

namespace
{
using Vector = Eigen::VectorXd;

struct Quadratic
{
    Vector target;

    double operator()(const Vector& values, Vector& gradient) const
    {
        gradient = values - target;
        return 0.5 * gradient.squaredNorm();
    }
};
}

int main()
{
    Eigen::VectorXi controls(3);
    controls << 20, 20, 50;
    Vector coefficients = Vector::LinSpaced(60000, -1.0, 1.0);
    Vector lower = Vector::Constant(60000, -100.0);
    Vector upper = Vector::Constant(60000, 100.0);
    const mfa::NSHardTraceBoundsAttestation attestation =
        mfa::ApplyNSHardTraceBounds(coefficients, controls, lower, upper);
    if (attestation.locked_u != 4124 || attestation.locked_v != 4124 ||
        attestation.locked_p != 0 || attestation.free_total != 51752 ||
        attestation.locked_indices.size() != 8248)
    {
        std::cerr << "hard-trace count or union drift\n";
        return 1;
    }
    for (size_t i = 0; i < attestation.locked_indices.size(); ++i)
    {
        const int index = attestation.locked_indices[i];
        if (lower(index) != coefficients(index) ||
            upper(index) != coefficients(index))
        {
            std::cerr << "hard-trace equality bound drift\n";
            return 1;
        }
    }
    const int field_size = 20 * 20 * 50;
    for (int field = 0; field < 3; ++field)
        for (int time = 0; time < 50; ++time)
            for (int y = 0; y < 20; ++y)
                for (int x = 0; x < 20; ++x)
                {
                    const int index = field * field_size +
                        x + 20 * (y + 20 * time);
                    const bool expected_locked = field < 2 &&
                        (x == 0 || x == 19 || y == 0 || y == 19 ||
                         time == 0);
                    const bool observed_locked =
                        lower(index) == coefficients(index) &&
                        upper(index) == coefficients(index);
                    if (observed_locked != expected_locked)
                    {
                        std::cerr << "hard-trace exhaustive mask drift\n";
                        return 1;
                    }
                    if (!expected_locked &&
                        (lower(index) != -100.0 || upper(index) != 100.0))
                    {
                        std::cerr << "hard-trace free bound drift\n";
                        return 1;
                    }
                }
    if ((lower.segment(40000, 20000).array() != -100.0).any() ||
        (upper.segment(40000, 20000).array() != 100.0).any())
    {
        std::cerr << "pressure coefficients were unexpectedly locked\n";
        return 1;
    }

    Vector gradient = Vector::Ones(60000);
    const double projected = mfa::NSProjectedGradientInfinity(
        coefficients, gradient, lower, upper);
    if (std::abs(projected - 1.0) > 1.0e-14)
    {
        std::cerr << "projected-gradient infinity norm drift\n";
        return 1;
    }
    Vector locked_only = Vector::Zero(60000);
    for (size_t i = 0; i < attestation.locked_indices.size(); ++i)
        locked_only(attestation.locked_indices[i]) = 3.0;
    if (mfa::NSProjectedGradientInfinity(
            coefficients, locked_only, lower, upper) != 0.0)
    {
        std::cerr << "equality-bound gradient was not projected to zero\n";
        return 1;
    }

    // Exercise LBFGS++ itself with one equality-bound coordinate.
    {
        Vector values(3);
        values << 0.5, 0.0, 0.0;
        const Vector initial = values;
        Vector small_lower(3), small_upper(3);
        small_lower << values(0), -100.0, -100.0;
        small_upper << values(0), 100.0, 100.0;
        Quadratic objective;
        objective.target.resize(3);
        objective.target << 9.0, -3.0, 4.0;
        LBFGSpp::LBFGSBParam<double> parameters;
        parameters.epsilon = 1.0e-12;
        parameters.epsilon_rel = 0.0;
        parameters.past = 0;
        parameters.max_iterations = 100;
        LBFGSpp::LBFGSBSolver<double> solver(parameters);
        double value = 0.0;
        solver.minimize(objective, values, value, small_lower, small_upper);
        if (std::abs(solver.initial_grad_norm() - 4.0) > 1.0e-14 ||
            values(0) != initial(0) || std::abs(values(1) + 3.0) > 1.0e-10 ||
            std::abs(values(2) - 4.0) > 1.0e-10 ||
            solver.update_count() < 1 ||
            solver.final_grad_norm() > parameters.epsilon)
        {
            std::cerr << "LBFGS++ equality-bound solve failed\n";
            return 1;
        }
    }

    {
        Vector nonfinite = coefficients;
        nonfinite(attestation.locked_indices.front()) =
            std::numeric_limits<double>::quiet_NaN();
        bool rejected = false;
        try
        {
            (void)mfa::NSHardTraceMaximumDrift(
                nonfinite, coefficients, attestation.locked_indices);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        if (!rejected)
        {
            std::cerr << "non-finite hard-trace drift was not rejected\n";
            return 1;
        }
    }

    return 0;
}
