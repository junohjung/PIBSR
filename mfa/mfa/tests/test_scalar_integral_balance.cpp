#include <mfa/scalar_integral_balance.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

typedef double Real;
typedef Eigen::VectorXd Vector;
typedef mfa::ScalarIntegralBalance<Real> Balance;
typedef mfa::ScalarIntegralBalanceConfig<Real> Config;
typedef mfa::ScalarIntegralBalanceOperators<Real> Operators;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(
    Real actual,
    Real expected,
    Real tolerance,
    const std::string& message)
{
    const Real scale = std::max<Real>(
        Real(1), std::max(std::abs(actual), std::abs(expected)));
    if (std::abs(actual - expected) > tolerance * scale)
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

std::vector<Real> openUniformKnots(int controls)
{
    require(controls >= 4, "cubic spline needs at least four controls");
    std::vector<Real> knots(static_cast<std::size_t>(controls + 4), Real(0));
    const int interior = controls - 4;
    for (int index = 0; index < interior; ++index)
        knots[static_cast<std::size_t>(4 + index)] =
            static_cast<Real>(index + 1) /
            static_cast<Real>(interior + 1);
    for (int index = controls; index < controls + 4; ++index)
        knots[static_cast<std::size_t>(index)] = Real(1);
    return knots;
}

Real greville(
    const std::vector<Real>& knots,
    int control,
    Real lower,
    Real upper)
{
    const Real parameter =
        (knots[static_cast<std::size_t>(control + 1)] +
         knots[static_cast<std::size_t>(control + 2)] +
         knots[static_cast<std::size_t>(control + 3)]) / Real(3);
    return lower + (upper - lower) * parameter;
}

Config makeConfig(int parts_x)
{
    Config config;
    config.lambda_balance = Real(1);
    config.parts_x = parts_x;
    config.fit_time_indices = std::vector<int>{1, 3, 5, 7};
    config.validation_time_indices = std::vector<int>{2, 4, 6, 8};
    return config;
}

Operators buildOperators(int parts_x, int quadrature_order)
{
    std::array<std::vector<Real>, 2> knots;
    knots[0] = openUniformKnots(8);
    knots[1] = openUniformKnots(7);
    const std::array<int, 2> degrees = {{3, 3}};
    const std::array<Real, 2> lower = {{0.0, 0.0}};
    const std::array<Real, 2> upper = {{1.0, 0.4}};
    std::vector<Real> time_axis(9);
    for (int index = 0; index < 9; ++index)
        time_axis[static_cast<std::size_t>(index)] = Real(0.05) * index;
    return Balance::build(
        knots, degrees, lower, upper, time_axis,
        makeConfig(parts_x), quadrature_order);
}

Vector manufacturedCoefficients(Real advection)
{
    const std::vector<Real> knots_x = openUniformKnots(8);
    const std::vector<Real> knots_t = openUniformKnots(7);
    Vector coefficients(8 * 7);
    for (int time = 0; time < 7; ++time)
        for (int x = 0; x < 8; ++x)
            coefficients(x + 8 * time) =
                greville(knots_x, x, 0.0, 1.0) -
                advection * greville(knots_t, time, 0.0, 0.4);
    return coefficients;
}

void testZeroConstantAndManufactured()
{
    const Operators operators = buildOperators(4, 4);
    const int coefficient_count = operators.fit.coefficientCount();

    const mfa::ScalarIntegralBalanceEvaluation<Real> zero = Balance::evaluate(
        operators.fit, Vector::Zero(coefficient_count), 1.0, 0.05, 1.0, true);
    require(zero.raw_balance.norm() == 0.0, "zero field residual is not exact");
    require(zero.coefficient_gradient.norm() == 0.0,
            "zero field gradient is not exact");

    const mfa::ScalarIntegralBalanceEvaluation<Real> constant = Balance::evaluate(
        operators.fit, Vector::Constant(coefficient_count, 2.5),
        1.0, 0.05, 1.0, false);
    require(constant.raw_balance.lpNorm<Eigen::Infinity>() < 2.0e-13,
            "constant field does not close");

    const Real advection = 0.73;
    const mfa::ScalarIntegralBalanceEvaluation<Real> manufactured =
        Balance::evaluate(
            operators.fit, manufacturedCoefficients(advection),
            advection, 0.05, 1.0, false);
    require(manufactured.raw_balance.lpNorm<Eigen::Infinity>() < 3.0e-13,
            "manufactured affine solution does not close");

    require(operators.fit.maximum_internal_face_closure == 0.0,
            "shared internal faces do not cancel algebraically");
    require(operators.fit.faceSampleCount() == 20,
            "q=4 did not construct one shared sample per face and time");
    for (Eigen::Index row = 0; row < operators.fit.scale.size(); ++row)
        requireNear(operators.fit.scale(row), 2.0, 1.0e-15,
                    "A_K^-1/2 scaling is incorrect");
}

Vector deterministicCoefficients(int count)
{
    Vector coefficients(count);
    for (int index = 0; index < count; ++index)
        coefficients(index) =
            std::sin(Real(index + 1) * Real(0.37)) +
            Real(0.2) * std::cos(Real(index + 1) * Real(0.11));
    return coefficients;
}

void testHierarchyAndQuadrature()
{
    const Operators q1_gl4 = buildOperators(1, 4);
    const Operators q4_gl4 = buildOperators(4, 4);
    const Operators q4_gl8 = buildOperators(4, 8);
    const Vector coefficients =
        deterministicCoefficients(q1_gl4.fit.coefficientCount());

    const mfa::ScalarIntegralBalanceEvaluation<Real> whole =
        Balance::evaluate(q1_gl4.fit, coefficients, 0.91, 0.05, 0.0, false);
    const mfa::ScalarIntegralBalanceEvaluation<Real> partitioned =
        Balance::evaluate(q4_gl4.fit, coefficients, 0.91, 0.05, 0.0, false);
    const mfa::ScalarIntegralBalanceEvaluation<Real> partitioned_gl8 =
        Balance::evaluate(q4_gl8.fit, coefficients, 0.91, 0.05, 0.0, false);

    for (int time = 0; time < 4; ++time)
        requireNear(
            partitioned.raw_balance.segment(time * 4, 4).sum(),
            whole.raw_balance(time), 2.0e-13,
            "partition hierarchy does not reproduce the whole-domain balance");
    require((partitioned.raw_balance - partitioned_gl8.raw_balance)
                .lpNorm<Eigen::Infinity>() < 3.0e-13,
            "GL4 and GL8 disagree for the cubic representation");
}

void testExactCoefficientGradientAndZeroWeight()
{
    const Operators operators = buildOperators(4, 4);
    const Vector coefficients =
        deterministicCoefficients(operators.fit.coefficientCount());
    Vector direction(coefficients.size());
    for (Eigen::Index index = 0; index < direction.size(); ++index)
        direction(index) = std::sin(Real(index + 1) * Real(0.6180339887498948));
    direction.normalize();

    const Real lambda = 1.7;
    const mfa::ScalarIntegralBalanceEvaluation<Real> evaluation =
        Balance::evaluate(
            operators.fit, coefficients, 1.0, 0.05, lambda, true);
    const Real step = 1.0e-6;
    const Real plus = Balance::evaluate(
        operators.fit, coefficients + step * direction,
        1.0, 0.05, lambda, false).weighted_loss;
    const Real minus = Balance::evaluate(
        operators.fit, coefficients - step * direction,
        1.0, 0.05, lambda, false).weighted_loss;
    const Real finite_difference = (plus - minus) / (2.0 * step);
    const Real analytic = evaluation.coefficient_gradient.dot(direction);
    requireNear(finite_difference, analytic, 2.0e-9,
                "coefficient-space directional gradient failed");

    const mfa::ScalarIntegralBalanceEvaluation<Real> zero_weight =
        Balance::evaluate(
            operators.fit, coefficients, 1.0, 0.05, 0.0, true);
    require(zero_weight.weighted_loss == 0.0,
            "zero integral weight produced a loss");
    require(zero_weight.coefficient_gradient.norm() == 0.0,
            "zero integral weight produced a gradient");
}

void testTimeRoleGuards()
{
    std::array<std::vector<Real>, 2> knots;
    knots[0] = openUniformKnots(8);
    knots[1] = openUniformKnots(7);
    const std::array<int, 2> degrees = {{3, 3}};
    const std::array<Real, 2> lower = {{0.0, 0.0}};
    const std::array<Real, 2> upper = {{1.0, 0.4}};
    std::vector<Real> time_axis(9);
    for (int index = 0; index < 9; ++index)
        time_axis[static_cast<std::size_t>(index)] = Real(0.05) * index;

    Config overlap = makeConfig(1);
    overlap.validation_time_indices[0] = overlap.fit_time_indices[0];
    bool overlap_rejected = false;
    try
    {
        Balance::build(knots, degrees, lower, upper, time_axis, overlap, 4);
    }
    catch (const std::invalid_argument&)
    {
        overlap_rejected = true;
    }
    require(overlap_rejected, "overlapping fit/validation roles were accepted");

    Config initial_time = makeConfig(1);
    initial_time.fit_time_indices[0] = 0;
    bool initial_rejected = false;
    try
    {
        Balance::build(knots, degrees, lower, upper, time_axis, initial_time, 4);
    }
    catch (const std::invalid_argument&)
    {
        initial_rejected = true;
    }
    require(initial_rejected, "initial-time role was accepted");
}

void testZeroWeightEnvironmentFastPath()
{
    ::setenv("MFA_LAMBDA_INT_BALANCE", "0", 1);
    ::setenv("MFA_INT_DIAGNOSTICS", "0", 1);
    ::setenv("MFA_INT_PARTS_X", "deliberately-not-an-integer", 1);
    ::setenv("MFA_INT_QUADRATURE", "deliberately-not-a-rule", 1);
    ::unsetenv("MFA_INT_FIT_TIME_INDEX_FILE");
    ::unsetenv("MFA_INT_VALID_TIME_INDEX_FILE");
    const Config config = Config::fromEnvironment();
    require(!config.operatorsRequired(),
            "zero-weight environment did not take the fast path");
    ::unsetenv("MFA_LAMBDA_INT_BALANCE");
    ::unsetenv("MFA_INT_DIAGNOSTICS");
    ::unsetenv("MFA_INT_PARTS_X");
    ::unsetenv("MFA_INT_QUADRATURE");
}

} // namespace

int main()
{
    try
    {
        testZeroConstantAndManufactured();
        testHierarchyAndQuadrature();
        testExactCoefficientGradientAndZeroWeight();
        testTimeRoleGuards();
        testZeroWeightEnvironmentFastPath();
        std::cout << "scalar_integral_balance_tests=PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "scalar_integral_balance_tests=FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
