#include <mfa/mfa.hpp>
#include <mfa/ns_knot_span_integral_balance.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef mfa::MFA_Data<Scalar> MFAData;
typedef mfa::NSKnotSpanGLBalance<Scalar> Balance;
typedef mfa::NSKnotSpanGLBalanceOperators<Scalar> Operators;
typedef mfa::NSKnotSpanGLBlockOperators<Scalar> BlockOperators;
typedef mfa::NSKnotSpanGLBasis1D<Scalar> GLBasis;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(
    Scalar actual,
    Scalar expected,
    Scalar tolerance,
    const std::string& message)
{
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance)
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

struct ProductionFixture
{
    Eigen::Vector3i degree;
    Eigen::Vector3i controls;
    MFAData data;
    std::array<std::vector<Scalar>, 3> knots;
    std::array<int, 3> degrees;
    std::array<Scalar, 3> lower;
    std::array<Scalar, 3> upper;
    std::vector<Scalar> times;
    mfa::NSIntegralBalanceConfig<Scalar> config;

    ProductionFixture()
        : degree(3, 3, 3),
          controls(20, 20, 50),
          data(degree, controls, 0, 0)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            knots[axis] = data.tmesh.all_knots[axis];
            degrees[axis] = degree(axis);
        }
        lower = {{0.0, 0.0, 0.0}};
        upper = {{1.0, 1.0, 10.0}};
        times.resize(1001);
        for (int index = 0; index <= 1000; ++index)
            times[index] = 0.01 * static_cast<Scalar>(index);
        config.lambda_mass = 1.0;
        config.lambda_xmom = 1.0;
        config.lambda_ymom = 1.0;
        config.parts_x = 4;
        config.parts_y = 4;
        config.physics_t_min = 0.01;
        config.viscosity = 0.001;
        config.check_closure = true;
        config.closure_tolerance = 1.0e-13;
        config.fit_time_indices = {1, 317, 799};
        config.validation_time_indices = {125, 999};
    }
};

void compareBasisSet(
    MFAData& data,
    int axis,
    const std::vector<GLBasis>& bases,
    Scalar& maximum_value_error,
    Scalar& maximum_derivative_error)
{
    const Eigen::Vector3i order = data.p + Eigen::Vector3i::Ones();
    mfa::BasisFunInfo<Scalar> scratch(order);
    std::vector<std::vector<Scalar> > derivatives(
        2, std::vector<Scalar>(4, 0.0));
    const TensorProduct<Scalar>& tensor =
        data.tmesh.tensor_prods[0];
    for (std::size_t index = 0; index < bases.size(); ++index)
    {
        const GLBasis& candidate = bases[index];
        const int span = data.tmesh.FindSpan(axis, candidate.parameter);
        data.FastBasisFunsDers(
            axis, candidate.parameter, span, 1, derivatives, scratch);
        const int expected_first =
            span - data.p(axis) - tensor.knot_mins[axis];
        require(
            candidate.first_control == expected_first,
            "GL/MFA first-control mismatch on axis " +
            std::to_string(axis));
        for (int local = 0; local < 4; ++local)
        {
            maximum_value_error = std::max(
                maximum_value_error,
                std::abs(candidate.value[local] - derivatives[0][local]));
            maximum_derivative_error = std::max(
                maximum_derivative_error,
                std::abs(
                    candidate.derivative[local] - derivatives[1][local]));
        }
    }
}

struct Jet
{
    Scalar value;
    Scalar dx;
    Scalar dy;
};

Jet operatorJet(
    const BlockOperators& operators,
    const mfa::NSKnotSpanGLFaceSample& sample,
    const Vector& coefficients)
{
    const GLBasis& x = operators.x_bases[sample.x_basis];
    const GLBasis& y = operators.y_bases[sample.y_basis];
    const GLBasis& t = operators.t_bases[sample.t_basis];
    Jet result = {0.0, 0.0, 0.0};
    for (int local_t = 0; local_t < 4; ++local_t)
        for (int local_y = 0; local_y < 4; ++local_y)
            for (int local_x = 0; local_x < 4; ++local_x)
            {
                const int control_x = x.first_control + local_x;
                const int control_y = y.first_control + local_y;
                const int control_t = t.first_control + local_t;
                const int coefficient_index = control_x + 20 *
                    (control_y + 20 * control_t);
                const Scalar coefficient = coefficients(coefficient_index);
                result.value += coefficient * x.value[local_x] *
                    y.value[local_y] * t.value[local_t];
                result.dx += coefficient * x.derivative[local_x] *
                    y.value[local_y] * t.value[local_t];
                result.dy += coefficient * x.value[local_x] *
                    y.derivative[local_y] * t.value[local_t];
            }
    result.dx /= operators.physical_extents[0];
    result.dy /= operators.physical_extents[1];
    return result;
}

Jet productionMFAJet(
    MFAData& data,
    const BlockOperators& operators,
    const mfa::NSKnotSpanGLFaceSample& sample,
    const Vector& coefficients)
{
    const Scalar parameters[] = {
        operators.x_bases[sample.x_basis].parameter,
        operators.y_bases[sample.y_basis].parameter,
        operators.t_bases[sample.t_basis].parameter};
    const Eigen::Vector3i order = data.p + Eigen::Vector3i::Ones();
    mfa::BasisFunInfo<Scalar> scratch(order);
    std::array<std::vector<std::vector<Scalar> >, 3> derivative;
    std::array<int, 3> first;
    const TensorProduct<Scalar>& tensor =
        data.tmesh.tensor_prods[0];
    for (int axis = 0; axis < 3; ++axis)
    {
        derivative[axis].assign(2, std::vector<Scalar>(4, 0.0));
        const int span = data.tmesh.FindSpan(axis, parameters[axis]);
        data.FastBasisFunsDers(
            axis, parameters[axis], span, 1,
            derivative[axis], scratch);
        first[axis] = span - data.p(axis) - tensor.knot_mins[axis];
    }

    // This is the exact stride/order used by Encoder::CollMatrixUnified:
    // x is fastest, then y, then time.
    const int strides[] = {1, 20, 20 * 20};
    Jet result = {0.0, 0.0, 0.0};
    for (int local_t = 0; local_t < 4; ++local_t)
        for (int local_y = 0; local_y < 4; ++local_y)
            for (int local_x = 0; local_x < 4; ++local_x)
            {
                const int local[] = {local_x, local_y, local_t};
                int coefficient_index = 0;
                for (int axis = 0; axis < 3; ++axis)
                    coefficient_index +=
                        (first[axis] + local[axis]) * strides[axis];
                const Scalar coefficient = coefficients(coefficient_index);
                result.value += coefficient *
                    derivative[0][0][local_x] *
                    derivative[1][0][local_y] *
                    derivative[2][0][local_t];
                result.dx += coefficient *
                    derivative[0][1][local_x] *
                    derivative[1][0][local_y] *
                    derivative[2][0][local_t];
                result.dy += coefficient *
                    derivative[0][0][local_x] *
                    derivative[1][1][local_y] *
                    derivative[2][0][local_t];
            }
    result.dx /= operators.physical_extents[0];
    result.dy /= operators.physical_extents[1];
    return result;
}

void testCoefficientOrdering(
    ProductionFixture& fixture,
    const Operators& operators)
{
    Vector coefficients(20 * 20 * 50);
    std::mt19937_64 generator(20260803u);
    std::normal_distribution<Scalar> distribution(0.0, 0.25);
    for (int index = 0; index < coefficients.size(); ++index)
        coefficients(index) = distribution(generator);

    Scalar maximum_value_error = 0.0;
    Scalar maximum_dx_error = 0.0;
    Scalar maximum_dy_error = 0.0;
    const std::size_t samples = operators.fit.face_samples.size();
    const std::size_t comparisons = std::min<std::size_t>(samples, 257);
    for (std::size_t comparison = 0;
         comparison < comparisons; ++comparison)
    {
        const std::size_t sample_index =
            comparisons == 1 ? 0 :
            comparison * (samples - 1) / (comparisons - 1);
        const mfa::NSKnotSpanGLFaceSample& sample =
            operators.fit.face_samples[sample_index];
        const Jet candidate = operatorJet(
            operators.fit, sample, coefficients);
        const Jet production = productionMFAJet(
            fixture.data, operators.fit, sample, coefficients);
        maximum_value_error = std::max(
            maximum_value_error,
            std::abs(candidate.value - production.value));
        maximum_dx_error = std::max(
            maximum_dx_error,
            std::abs(candidate.dx - production.dx));
        maximum_dy_error = std::max(
            maximum_dy_error,
            std::abs(candidate.dy - production.dy));
    }
    require(maximum_value_error < 2.0e-14, "MFA value ordering mismatch");
    require(maximum_dx_error < 3.0e-13, "MFA x-derivative ordering mismatch");
    require(maximum_dy_error < 3.0e-13, "MFA y-derivative ordering mismatch");
    std::cout
        << "NS_GL_MFA_ORDER status=pass"
        << " compared_samples=" << comparisons
        << " max_value_error=" << maximum_value_error
        << " max_dx_error=" << maximum_dx_error
        << " max_dy_error=" << maximum_dy_error << "\n";
}

} // anonymous namespace

int main()
{
    try
    {
        ProductionFixture fixture;
        const Operators operators = Balance::build(
            fixture.knots, fixture.degrees,
            fixture.lower, fixture.upper, fixture.times,
            fixture.config, 4);
        Scalar maximum_value_error = 0.0;
        Scalar maximum_derivative_error = 0.0;
        compareBasisSet(
            fixture.data, 0, operators.fit.x_bases,
            maximum_value_error, maximum_derivative_error);
        compareBasisSet(
            fixture.data, 1, operators.fit.y_bases,
            maximum_value_error, maximum_derivative_error);
        compareBasisSet(
            fixture.data, 2, operators.fit.t_bases,
            maximum_value_error, maximum_derivative_error);
        compareBasisSet(
            fixture.data, 0, operators.validation.x_bases,
            maximum_value_error, maximum_derivative_error);
        compareBasisSet(
            fixture.data, 1, operators.validation.y_bases,
            maximum_value_error, maximum_derivative_error);
        compareBasisSet(
            fixture.data, 2, operators.validation.t_bases,
            maximum_value_error, maximum_derivative_error);
        require(
            maximum_value_error < 2.0e-15,
            "GL basis values differ from production MFA");
        require(
            maximum_derivative_error < 4.0e-14,
            "GL basis derivatives differ from production MFA");
        std::cout
            << "NS_GL_MFA_BASIS status=pass"
            << " layout=20x20x50 degree=3"
            << " max_value_error=" << maximum_value_error
            << " max_param_derivative_error=" << maximum_derivative_error
            << "\n";
        testCoefficientOrdering(fixture, operators);
        std::cout
            << "NS_GL_MFA_COMPATIBILITY status=pass"
            << " coefficient_order=x_fast_y_then_time\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "NS_GL_MFA_COMPATIBILITY status=fail error="
            << error.what() << "\n";
        return 1;
    }
}
