#include <mfa/ns_integral_balance.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef mfa::NSIntegralBalance<Scalar> Balance;
typedef mfa::NSIntegralBalanceConfig<Scalar> Config;
typedef mfa::NSIntegralBalanceOperators<Scalar> Operators;
typedef mfa::NSIntegralBalanceEvaluation<Scalar> Evaluation;

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
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance)
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

Matrix cavityCoordinates()
{
    const int nx = 21;
    const int ny = 21;
    const int nt = 1001;
    Matrix coordinates(nx * ny * nt, 3);
    int row = 0;
    for (int it = 0; it < nt; ++it)
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix, ++row)
            {
                coordinates(row, 0) = static_cast<Scalar>(ix) / 20.0;
                coordinates(row, 1) = static_cast<Scalar>(iy) / 20.0;
                coordinates(row, 2) = static_cast<Scalar>(it) / 100.0;
            }
    return coordinates;
}

Config baseConfig(int parts_x, int parts_y)
{
    Config config;
    config.lambda_mass = 1.0;
    config.lambda_xmom = 1.0;
    config.lambda_ymom = 1.0;
    config.parts_x = parts_x;
    config.parts_y = parts_y;
    config.physics_t_min = 0.01;
    config.fit_time_indices.push_back(1);
    config.fit_time_indices.push_back(500);
    config.validation_time_indices.push_back(1000);
    return config;
}

Evaluation evaluate(
    const mfa::NSIntegralBlockOperators<Scalar>& op,
    const Vector& u,
    const Vector& ut,
    const Vector& ux,
    const Vector& uy,
    const Vector& v,
    const Vector& vt,
    const Vector& vx,
    const Vector& vy,
    const Vector& p,
    Scalar mass_weight,
    Scalar x_weight,
    Scalar y_weight,
    bool adjoints = false)
{
    return Balance::evaluate(
        op, u, ut, ux, uy, v, vt, vx, vy, p,
        0.001, mass_weight, x_weight, y_weight, adjoints);
}

struct Fields
{
    Vector u, ut, ux, uy, v, vt, vx, vy, p;
};

Fields zeroFields(int count)
{
    Fields fields;
    fields.u = Vector::Zero(count);
    fields.ut = Vector::Zero(count);
    fields.ux = Vector::Zero(count);
    fields.uy = Vector::Zero(count);
    fields.v = Vector::Zero(count);
    fields.vt = Vector::Zero(count);
    fields.vx = Vector::Zero(count);
    fields.vy = Vector::Zero(count);
    fields.p = Vector::Zero(count);
    return fields;
}

Evaluation evaluate(
    const mfa::NSIntegralBlockOperators<Scalar>& op,
    const Fields& fields,
    Scalar mass_weight,
    Scalar x_weight,
    Scalar y_weight,
    bool adjoints = false)
{
    return evaluate(
        op,
        fields.u, fields.ut, fields.ux, fields.uy,
        fields.v, fields.vt, fields.vx, fields.vy, fields.p,
        mass_weight, x_weight, y_weight, adjoints);
}

Fields shifted(const Fields& base, const Fields& direction, Scalar step)
{
    Fields result;
    result.u = base.u + step * direction.u;
    result.ut = base.ut + step * direction.ut;
    result.ux = base.ux + step * direction.ux;
    result.uy = base.uy + step * direction.uy;
    result.v = base.v + step * direction.v;
    result.vt = base.vt + step * direction.vt;
    result.vx = base.vx + step * direction.vx;
    result.vy = base.vy + step * direction.vy;
    result.p = base.p + step * direction.p;
    return result;
}

Scalar combinedNorm(const Fields& fields)
{
    return std::sqrt(
        fields.u.squaredNorm() + fields.ut.squaredNorm() +
        fields.ux.squaredNorm() + fields.uy.squaredNorm() +
        fields.v.squaredNorm() + fields.vt.squaredNorm() +
        fields.vx.squaredNorm() + fields.vy.squaredNorm() +
        fields.p.squaredNorm());
}

void scale(Fields& fields, Scalar factor)
{
    fields.u *= factor;
    fields.ut *= factor;
    fields.ux *= factor;
    fields.uy *= factor;
    fields.v *= factor;
    fields.vt *= factor;
    fields.vx *= factor;
    fields.vy *= factor;
    fields.p *= factor;
}

Scalar adjointDot(const Evaluation& evaluation, const Fields& direction)
{
    return
        evaluation.adjoints.u.dot(direction.u) +
        evaluation.adjoints.ut.dot(direction.ut) +
        evaluation.adjoints.ux.dot(direction.ux) +
        evaluation.adjoints.uy.dot(direction.uy) +
        evaluation.adjoints.v.dot(direction.v) +
        evaluation.adjoints.vt.dot(direction.vt) +
        evaluation.adjoints.vx.dot(direction.vx) +
        evaluation.adjoints.vy.dot(direction.vy) +
        evaluation.adjoints.p.dot(direction.p);
}

void fillNormal(Vector& vector, std::mt19937_64& generator, Scalar scale_value)
{
    std::normal_distribution<Scalar> normal(0.0, scale_value);
    for (int i = 0; i < vector.size(); ++i)
        vector(i) = normal(generator);
}

void testManufacturedAndClosure(
    const Matrix& coordinates,
    const Operators& four_by_four)
{
    const int count = static_cast<int>(coordinates.rows());
    Fields fields = zeroFields(count);

    Evaluation result = evaluate(
        four_by_four.fit, fields, 1.0, 1.0, 1.0);
    require(result.raw_mass.isZero(0.0), "zero-field mass is not exact zero");
    require(result.raw_xmom.isZero(0.0), "zero-field x momentum is not exact zero");
    require(result.raw_ymom.isZero(0.0), "zero-field y momentum is not exact zero");

    fields.u.setConstant(0.7);
    fields.v.setConstant(-0.2);
    fields.p.setConstant(4.0);
    result = evaluate(four_by_four.fit, fields, 1.0, 1.0, 1.0);
    require(
        result.raw_mass.lpNorm<Eigen::Infinity>() < 1.0e-14,
        "constant-field mass balance failed");
    require(
        result.raw_xmom.lpNorm<Eigen::Infinity>() < 1.0e-13,
        "constant-field x momentum balance failed");
    require(
        result.raw_ymom.lpNorm<Eigen::Infinity>() < 1.0e-13,
        "constant-field y momentum balance failed");
    const Vector x_before = result.raw_xmom;
    const Vector y_before = result.raw_ymom;
    fields.p.array() += 123.0;
    result = evaluate(four_by_four.fit, fields, 1.0, 1.0, 1.0);
    require(
        (result.raw_xmom - x_before).lpNorm<Eigen::Infinity>() < 1.0e-12,
        "constant pressure shift changed x momentum");
    require(
        (result.raw_ymom - y_before).lpNorm<Eigen::Infinity>() < 1.0e-12,
        "constant pressure shift changed y momentum");

    const Scalar acceleration = 0.35;
    fields = zeroFields(count);
    fields.u = acceleration * coordinates.col(2);
    fields.ut.setConstant(acceleration);
    fields.p = -acceleration * coordinates.col(0);
    result = evaluate(four_by_four.fit, fields, 1.0, 1.0, 1.0);
    require(
        result.raw_xmom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "u=a*t, p=-a*x transient/pressure cancellation failed");

    // Plane Poiseuille manufactured field:
    // u=C*y*(1-y), p=-2*nu*C*x.  The pressure and viscous face
    // fluxes cancel on every subdomain independently.
    const Scalar profile_scale = 0.8;
    const Scalar viscosity = 0.001;
    fields = zeroFields(count);
    fields.u =
        profile_scale *
        coordinates.col(1).array() *
        (1.0 - coordinates.col(1).array());
    fields.uy =
        profile_scale * (1.0 - 2.0 * coordinates.col(1).array());
    fields.p =
        -2.0 * viscosity * profile_scale * coordinates.col(0);
    result = evaluate(four_by_four.fit, fields, 1.0, 1.0, 1.0);
    require(
        result.raw_xmom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "Poiseuille pressure/viscous flux cancellation failed");
    require(
        result.raw_ymom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "Poiseuille transverse balance failed");

    Config global_config = baseConfig(1, 1);
    Operators global = Balance::build(coordinates, global_config);
    std::mt19937_64 generator(20260729u);
    fields = zeroFields(count);
    fillNormal(fields.u, generator, 0.4);
    fillNormal(fields.v, generator, 0.3);
    fillNormal(fields.ux, generator, 0.2);
    fillNormal(fields.uy, generator, 0.2);
    fillNormal(fields.vx, generator, 0.2);
    fillNormal(fields.vy, generator, 0.2);
    fillNormal(fields.p, generator, 0.5);

    const int blocks = 16;
    // Mass closure.
    Evaluation local = evaluate(
        four_by_four.fit, fields, 1.0, 0.0, 0.0);
    Evaluation whole = evaluate(
        global.fit, fields, 1.0, 0.0, 0.0);
    for (int time = 0; time < 2; ++time)
        requireNear(
            local.raw_mass.segment(time * blocks, blocks).sum(),
            whole.raw_mass(time), 2.0e-13,
            "random mass face closure failed");

    // Volume/time-term closure.
    Fields component = zeroFields(count);
    fillNormal(component.ut, generator, 0.3);
    fillNormal(component.vt, generator, 0.3);
    local = evaluate(four_by_four.fit, component, 0.0, 1.0, 1.0);
    whole = evaluate(global.fit, component, 0.0, 1.0, 1.0);
    for (int time = 0; time < 2; ++time)
    {
        requireNear(
            local.raw_xmom.segment(time * blocks, blocks).sum(),
            whole.raw_xmom(time), 2.0e-13,
            "random x volume/time closure failed");
        requireNear(
            local.raw_ymom.segment(time * blocks, blocks).sum(),
            whole.raw_ymom(time), 2.0e-13,
            "random y volume/time closure failed");
    }

    // Pressure-only closure.
    component = zeroFields(count);
    component.p = fields.p;
    local = evaluate(four_by_four.fit, component, 0.0, 1.0, 1.0);
    whole = evaluate(global.fit, component, 0.0, 1.0, 1.0);
    for (int time = 0; time < 2; ++time)
    {
        requireNear(
            local.raw_xmom.segment(time * blocks, blocks).sum(),
            whole.raw_xmom(time), 2.0e-13,
            "random pressure x-face closure failed");
        requireNear(
            local.raw_ymom.segment(time * blocks, blocks).sum(),
            whole.raw_ymom(time), 2.0e-13,
            "random pressure y-face closure failed");
    }

    // Advective-only closure.
    component = zeroFields(count);
    component.u = fields.u;
    component.v = fields.v;
    local = evaluate(four_by_four.fit, component, 0.0, 1.0, 1.0);
    whole = evaluate(global.fit, component, 0.0, 1.0, 1.0);
    for (int time = 0; time < 2; ++time)
    {
        requireNear(
            local.raw_xmom.segment(time * blocks, blocks).sum(),
            whole.raw_xmom(time), 2.0e-13,
            "random advective x-face closure failed");
        requireNear(
            local.raw_ymom.segment(time * blocks, blocks).sum(),
            whole.raw_ymom(time), 2.0e-13,
            "random advective y-face closure failed");
    }

    // Viscous-only closure.
    component = zeroFields(count);
    component.ux = fields.ux;
    component.uy = fields.uy;
    component.vx = fields.vx;
    component.vy = fields.vy;
    local = evaluate(four_by_four.fit, component, 0.0, 1.0, 1.0);
    whole = evaluate(global.fit, component, 0.0, 1.0, 1.0);
    for (int time = 0; time < 2; ++time)
    {
        requireNear(
            local.raw_xmom.segment(time * blocks, blocks).sum(),
            whole.raw_xmom(time), 2.0e-13,
            "random viscous x-face closure failed");
        requireNear(
            local.raw_ymom.segment(time * blocks, blocks).sum(),
            whole.raw_ymom(time), 2.0e-13,
            "random viscous y-face closure failed");
    }
    std::cout
        << "NS_INT_MANUFACTURED status=pass"
        << " cases=zero,constant,pressure_shift,transient_pressure,poiseuille"
        << " closure=mass,volume_time,pressure,advective,viscous\n";
}

void testCountsAndFailures(
    const Matrix& coordinates,
    const Operators& four_by_four)
{
    require(four_by_four.fit.rows() == 32, "4x4 fit row count mismatch");
    require(four_by_four.validation.rows() == 16, "4x4 validation row count mismatch");
    require(four_by_four.fit.V.nonZeros() == 2 * 16 * 36, "4x4 volume nnz mismatch");
    require(four_by_four.fit.Ex.nonZeros() == 2 * 16 * 12, "4x4 Ex nnz mismatch");
    require(four_by_four.fit.Ey.nonZeros() == 2 * 16 * 12, "4x4 Ey nnz mismatch");

    Config two_config = baseConfig(2, 2);
    Operators two = Balance::build(coordinates, two_config);
    require(two.fit.rows() == 8, "2x2 fit row count mismatch");
    require(two.fit.V.nonZeros() == 2 * 4 * 121, "2x2 volume nnz mismatch");
    require(two.fit.Ex.nonZeros() == 2 * 4 * 22, "2x2 Ex nnz mismatch");
    require(two.fit.Ey.nonZeros() == 2 * 4 * 22, "2x2 Ey nnz mismatch");

    Vector bc_times(42 * 1001);
    int row = 0;
    for (int it = 0; it <= 1000; ++it)
        for (int boundary = 0; boundary < 42; ++boundary, ++row)
            bc_times(row) = static_cast<Scalar>(
                static_cast<float>(0.01f * it));
    Scalar active_bc = 0.0;
    const Vector mask =
        mfa::nsPhysicsTimeMask<Scalar>(bc_times, 0.01, active_bc);
    require(active_bc == 42000.0, "t_min=0.01 BC active count mismatch");
    require(mask.head(42).isZero(0.0), "t=0 BC rows were not excluded");

    Config diagnostics = baseConfig(4, 4);
    diagnostics.lambda_mass = 0.0;
    diagnostics.lambda_xmom = 0.0;
    diagnostics.lambda_ymom = 0.0;
    diagnostics.diagnostics = true;
    require(!diagnostics.enabled(), "diagnostic-only config enabled objective");
    require(
        diagnostics.operatorsRequired(),
        "diagnostic-only config did not request operators");
    const Operators diagnostic_operators =
        Balance::build(coordinates, diagnostics);
    Fields diagnostic_fields = zeroFields(
        static_cast<int>(coordinates.rows()));
    diagnostic_fields.u = coordinates.col(0);
    diagnostic_fields.v = coordinates.col(1);
    const Evaluation diagnostic_result = evaluate(
        diagnostic_operators.validation,
        diagnostic_fields, 0.0, 0.0, 0.0, true);
    require(
        diagnostic_result.mse_mass > 0.0,
        "diagnostic-only evaluation did not measure a nonzero residual");
    require(
        diagnostic_result.weighted_loss == 0.0,
        "diagnostic-only evaluation changed the objective");
    require(
        diagnostic_result.adjoints.u.isZero(0.0) &&
        diagnostic_result.adjoints.ut.isZero(0.0) &&
        diagnostic_result.adjoints.ux.isZero(0.0) &&
        diagnostic_result.adjoints.uy.isZero(0.0) &&
        diagnostic_result.adjoints.v.isZero(0.0) &&
        diagnostic_result.adjoints.vt.isZero(0.0) &&
        diagnostic_result.adjoints.vx.isZero(0.0) &&
        diagnostic_result.adjoints.vy.isZero(0.0) &&
        diagnostic_result.adjoints.p.isZero(0.0),
        "diagnostic-only evaluation produced a nonzero adjoint");

    Vector ic_times = Vector::Zero(441);
    Scalar active_ic = 0.0;
    mfa::nsPhysicsTimeMask<Scalar>(ic_times, 0.0, active_ic);
    require(active_ic == 441.0, "historical IC role was not retained");

    bool failed = false;
    try
    {
        Config malformed = baseConfig(3, 4);
        Balance::build(coordinates, malformed);
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    require(failed, "nondivisible partition did not fail closed");

    failed = false;
    try
    {
        Config malformed = baseConfig(4, 4);
        malformed.fit_time_indices[0] = 0;
        Balance::build(coordinates, malformed);
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    require(failed, "t=0 integral row did not fail at t_min=0.01");

    failed = false;
    try
    {
        Matrix malformed = coordinates;
        malformed.row(1) = malformed.row(0);
        Balance::build(malformed, baseConfig(4, 4));
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    require(failed, "duplicate/missing grid point did not fail closed");

    failed = false;
    try
    {
        Matrix malformed = coordinates;
        // Preserve a structured 21-value axis but move one entire x plane,
        // so the frozen-spacing assertion, rather than duplicate detection,
        // is the expected failure.
        for (int row_index = 0; row_index < malformed.rows(); ++row_index)
            if (std::abs(malformed(row_index, 0) - 0.5) < 1.0e-14)
                malformed(row_index, 0) = 0.5002;
        Balance::build(malformed, baseConfig(4, 4));
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    require(failed, "off-protocol frozen x axis did not fail closed");
    std::cout
        << "NS_INT_STRUCTURE status=pass"
        << " rows_4x4=32 rows_2x2=8"
        << " bc_active_tmin_0.01=42000 ic_active=441"
        << " diagnostic_only=objective_and_adjoint_zero"
        << " fail_closed=partition,time,grid,axis\n";
}

void testMultiEpsilonGradients(
    const Operators& operators,
    int point_count)
{
    std::mt19937_64 generator(987654321u);
    Fields base = zeroFields(point_count);
    fillNormal(base.u, generator, 0.25);
    fillNormal(base.ut, generator, 0.15);
    fillNormal(base.ux, generator, 0.2);
    fillNormal(base.uy, generator, 0.2);
    fillNormal(base.v, generator, 0.25);
    fillNormal(base.vt, generator, 0.15);
    fillNormal(base.vx, generator, 0.2);
    fillNormal(base.vy, generator, 0.2);
    fillNormal(base.p, generator, 0.3);

    const Scalar weights[][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
    };
    const char* labels[] = {
        "mass_only", "xmom_only", "ymom_only", "all_integral"
    };
    const Scalar relative_steps[] = {
        1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7
    };
    const Scalar base_scale = std::max<Scalar>(1.0, combinedNorm(base));

    for (int component = 0; component < 4; ++component)
    {
        const Evaluation analytical = evaluate(
            operators.fit, base,
            weights[component][0],
            weights[component][1],
            weights[component][2],
            true);
        for (int direction_index = 0; direction_index < 5; ++direction_index)
        {
            Fields direction = zeroFields(point_count);
            fillNormal(direction.u, generator, 1.0);
            fillNormal(direction.ut, generator, 1.0);
            fillNormal(direction.ux, generator, 1.0);
            fillNormal(direction.uy, generator, 1.0);
            fillNormal(direction.v, generator, 1.0);
            fillNormal(direction.vt, generator, 1.0);
            fillNormal(direction.vx, generator, 1.0);
            fillNormal(direction.vy, generator, 1.0);
            fillNormal(direction.p, generator, 1.0);
            scale(direction, 1.0 / combinedNorm(direction));
            const Scalar derivative = adjointDot(analytical, direction);
            Scalar minimum_error = std::numeric_limits<Scalar>::infinity();
            std::vector<Scalar> remainders;
            for (int step_index = 0; step_index < 6; ++step_index)
            {
                const Scalar step =
                    relative_steps[step_index] * base_scale;
                const Scalar plus = evaluate(
                    operators.fit, shifted(base, direction, step),
                    weights[component][0],
                    weights[component][1],
                    weights[component][2]).weighted_loss;
                const Scalar minus = evaluate(
                    operators.fit, shifted(base, direction, -step),
                    weights[component][0],
                    weights[component][1],
                    weights[component][2]).weighted_loss;
                const Scalar finite_difference =
                    (plus - minus) / (2.0 * step);
                const Scalar denominator = std::max<Scalar>(
                    1.0, std::max(
                        std::abs(finite_difference),
                        std::abs(derivative)));
                const Scalar error =
                    std::abs(finite_difference - derivative) / denominator;
                minimum_error = std::min(minimum_error, error);
                remainders.push_back(std::abs(
                    plus - analytical.weighted_loss - step * derivative));
                std::cout
                    << "NS_INT_GRADIENT component=" << labels[component]
                    << " direction=" << direction_index
                    << " relative_step=" << relative_steps[step_index]
                    << " relative_error=" << error
                    << " taylor_remainder=" << remainders.back()
                    << "\n";
            }
            bool second_order = false;
            for (int i = 1; i < 5; ++i)
            {
                if (remainders[i] > 1.0e-15 &&
                    remainders[i - 1] > 0.0)
                {
                    const Scalar slope =
                        std::log(remainders[i - 1] / remainders[i]) /
                        std::log(relative_steps[i - 1] / relative_steps[i]);
                    if (std::isfinite(slope) &&
                        slope >= 1.7 && slope <= 2.3)
                        second_order = true;
                }
            }
            require(
                minimum_error < 1.0e-6,
                std::string("multi-epsilon gradient error failed for ") +
                labels[component]);
            require(
                second_order,
                std::string("Taylor slope failed for ") + labels[component]);
        }
    }
}

} // anonymous namespace

int main()
{
    try
    {
        const Matrix coordinates = cavityCoordinates();
        const Operators operators =
            Balance::build(coordinates, baseConfig(4, 4));
        testManufacturedAndClosure(coordinates, operators);
        testCountsAndFailures(coordinates, operators);
        testMultiEpsilonGradients(
            operators, static_cast<int>(coordinates.rows()));
        std::cout << "NS_INTEGRAL_BALANCE_TEST status=pass\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "NS_INTEGRAL_BALANCE_TEST status=fail error=\""
            << error.what() << "\"\n";
        return EXIT_FAILURE;
    }
}
