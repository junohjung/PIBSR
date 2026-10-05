#include <mfa/ns_knot_span_integral_balance.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
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
typedef mfa::NSIntegralBalanceConfig<Scalar> Config;
typedef mfa::NSKnotSpanGLBalance<Scalar> Balance;
typedef mfa::NSKnotSpanGLBalanceOperators<Scalar> Operators;
typedef mfa::NSKnotSpanGLBalanceEvaluation<Scalar> Evaluation;

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

std::vector<Scalar> openUniformKnots(int control_count)
{
    const int degree = 3;
    std::vector<Scalar> knots(control_count + degree + 1, 0.0);
    for (int index = control_count; index < static_cast<int>(knots.size()); ++index)
        knots[index] = 1.0;
    const int spans = control_count - degree;
    for (int interior = 1; interior < spans; ++interior)
        knots[degree + interior] =
            static_cast<Scalar>(interior) / static_cast<Scalar>(spans);
    return knots;
}

Scalar greville(const std::vector<Scalar>& knots, int control)
{
    return (knots[control + 1] +
            knots[control + 2] +
            knots[control + 3]) / 3.0;
}

struct Fixture
{
    std::array<std::vector<Scalar>, 3> knots;
    std::array<int, 3> degrees;
    std::array<Scalar, 3> lower;
    std::array<Scalar, 3> upper;
    std::vector<Scalar> times;
    Config config;

    Fixture()
    {
        knots[0] = openUniformKnots(7);
        knots[1] = openUniformKnots(6);
        knots[2] = openUniformKnots(6);
        degrees = {{3, 3, 3}};
        lower = {{-0.5, 0.25, 0.0}};
        upper = {{1.5, 1.75, 2.0}};
        times = {0.0, 0.2, 0.55, 1.1, 1.6, 2.0};
        config.lambda_mass = 1.0;
        config.lambda_xmom = 1.0;
        config.lambda_ymom = 1.0;
        config.parts_x = 2;
        config.parts_y = 2;
        config.physics_t_min = 0.1;
        config.viscosity = 0.03;
        config.check_closure = true;
        config.closure_tolerance = 1.0e-13;
        config.fit_time_indices = {1, 3, 4};
        config.validation_time_indices = {2, 5};
    }

    int nx() const { return static_cast<int>(knots[0].size()) - 4; }
    int ny() const { return static_cast<int>(knots[1].size()) - 4; }
    int nt() const { return static_cast<int>(knots[2].size()) - 4; }
    int count() const { return nx() * ny() * nt(); }
    int index(int x, int y, int t) const
    {
        return x + nx() * (y + ny() * t);
    }
};

Evaluation evaluate(
    const mfa::NSKnotSpanGLBlockOperators<Scalar>& operators,
    const Vector& u,
    const Vector& v,
    const Vector& p,
    Scalar mass_weight,
    Scalar x_weight,
    Scalar y_weight,
    bool adjoints = false)
{
    return Balance::evaluate(
        operators, u, v, p, 0.03,
        mass_weight, x_weight, y_weight, adjoints);
}

void fillNormal(Vector& values, std::mt19937_64& generator, Scalar scale)
{
    std::normal_distribution<Scalar> distribution(0.0, scale);
    for (int index = 0; index < values.size(); ++index)
        values(index) = distribution(generator);
}

Scalar gradientDot(
    const Evaluation& evaluation,
    const Vector& du,
    const Vector& dv,
    const Vector& dp)
{
    return evaluation.coefficient_adjoints.u.dot(du) +
        evaluation.coefficient_adjoints.v.dot(dv) +
        evaluation.coefficient_adjoints.p.dot(dp);
}

void testManufacturedIdentity(const Fixture& fixture, const Operators& gl4)
{
    Vector u = Vector::Zero(fixture.count());
    Vector v = Vector::Zero(fixture.count());
    Vector p = Vector::Zero(fixture.count());
    Evaluation result = evaluate(gl4.fit, u, v, p, 1.0, 1.0, 1.0);
    require(result.raw_mass.isZero(0.0), "zero-field mass is not exact zero");
    require(result.raw_xmom.isZero(0.0), "zero-field x momentum is not exact zero");
    require(result.raw_ymom.isZero(0.0), "zero-field y momentum is not exact zero");

    u.setConstant(0.7);
    v.setConstant(-0.2);
    p.setConstant(4.0);
    result = evaluate(gl4.fit, u, v, p, 1.0, 1.0, 1.0);
    require(
        result.raw_mass.lpNorm<Eigen::Infinity>() < 2.0e-14,
        "constant-field mass balance failed");
    require(
        result.raw_xmom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "constant-field x momentum balance failed");
    require(
        result.raw_ymom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "constant-field y momentum balance failed");

    const Vector x_before = result.raw_xmom;
    const Vector y_before = result.raw_ymom;
    p.array() += 123.0;
    result = evaluate(gl4.fit, u, v, p, 1.0, 1.0, 1.0);
    require(
        (result.raw_xmom - x_before).lpNorm<Eigen::Infinity>() < 2.0e-12,
        "constant pressure shift changed x momentum");
    require(
        (result.raw_ymom - y_before).lpNorm<Eigen::Infinity>() < 2.0e-12,
        "constant pressure shift changed y momentum");

    // Manufactured transient: u=a*t and p=-a*x.  The integrated storage
    // and pressure flux cancel on every finite region.
    const Scalar acceleration = 0.35;
    u.setZero();
    v.setZero();
    p.setZero();
    for (int kt = 0; kt < fixture.nt(); ++kt)
        for (int ky = 0; ky < fixture.ny(); ++ky)
            for (int kx = 0; kx < fixture.nx(); ++kx)
            {
                const Scalar physical_t = fixture.lower[2] +
                    (fixture.upper[2] - fixture.lower[2]) *
                    greville(fixture.knots[2], kt);
                const Scalar physical_x = fixture.lower[0] +
                    (fixture.upper[0] - fixture.lower[0]) *
                    greville(fixture.knots[0], kx);
                const int index = fixture.index(kx, ky, kt);
                u(index) = acceleration * physical_t;
                p(index) = -acceleration * physical_x;
            }
    result = evaluate(gl4.fit, u, v, p, 1.0, 1.0, 1.0);
    require(
        result.raw_mass.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "manufactured transient mass balance failed");
    require(
        result.raw_xmom.lpNorm<Eigen::Infinity>() < 2.0e-12,
        "manufactured transient/pressure balance failed");
    require(
        result.raw_ymom.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "manufactured transient transverse balance failed");

    require(
        gl4.fit.maximum_internal_face_closure == 0.0,
        "fit internal faces are not algebraically closed");
    require(
        gl4.validation.maximum_internal_face_closure == 0.0,
        "validation internal faces are not algebraically closed");
    std::cout
        << "NS_GL_MANUFACTURED status=pass"
        << " cases=zero,constant,pressure_shift,transient_pressure"
        << " internal_face_closure=exact\n";
}

void testGLAgreementAndHierarchy(
    const Fixture& fixture,
    const Operators& gl4,
    const Operators& gl8)
{
    std::mt19937_64 generator(20260803u);
    Vector u(fixture.count()), v(fixture.count()), p(fixture.count());
    fillNormal(u, generator, 0.4);
    fillNormal(v, generator, 0.3);
    fillNormal(p, generator, 0.5);
    const Evaluation four = evaluate(gl4.fit, u, v, p, 1.0, 1.0, 1.0);
    const Evaluation eight = evaluate(gl8.fit, u, v, p, 1.0, 1.0, 1.0);
    const Scalar agreement = std::max({
        (four.raw_mass - eight.raw_mass).lpNorm<Eigen::Infinity>(),
        (four.raw_xmom - eight.raw_xmom).lpNorm<Eigen::Infinity>(),
        (four.raw_ymom - eight.raw_ymom).lpNorm<Eigen::Infinity>()});
    require(agreement < 3.0e-12, "GL4/GL8 residual agreement failed");

    Config global_config = fixture.config;
    global_config.parts_x = 1;
    global_config.parts_y = 1;
    const Operators global = Balance::build(
        fixture.knots, fixture.degrees, fixture.lower, fixture.upper,
        fixture.times, global_config, 8);
    const Evaluation local = evaluate(gl8.fit, u, v, p, 1.0, 1.0, 1.0);
    const Evaluation whole = evaluate(global.fit, u, v, p, 1.0, 1.0, 1.0);
    const int blocks = fixture.config.parts_x * fixture.config.parts_y;
    for (int time = 0;
         time < static_cast<int>(fixture.config.fit_time_indices.size());
         ++time)
    {
        requireNear(
            local.raw_mass.segment(time * blocks, blocks).sum(),
            whole.raw_mass(time), 3.0e-12,
            "GL hierarchy mass closure failed");
        requireNear(
            local.raw_xmom.segment(time * blocks, blocks).sum(),
            whole.raw_xmom(time), 3.0e-12,
            "GL hierarchy x-momentum closure failed");
        requireNear(
            local.raw_ymom.segment(time * blocks, blocks).sum(),
            whole.raw_ymom(time), 3.0e-12,
            "GL hierarchy y-momentum closure failed");
    }
    std::cout
        << "NS_GL_AGREEMENT status=pass"
        << " max_gl4_gl8=" << agreement
        << " hierarchy=q2_to_q1\n";
}

void testAdjointAndTaylor(const Fixture& fixture, const Operators& gl4)
{
    std::mt19937_64 generator(991827u);
    Vector u(fixture.count()), v(fixture.count()), p(fixture.count());
    Vector du(fixture.count()), dv(fixture.count()), dp(fixture.count());
    fillNormal(u, generator, 0.2);
    fillNormal(v, generator, 0.2);
    fillNormal(p, generator, 0.25);
    fillNormal(du, generator, 1.0);
    fillNormal(dv, generator, 1.0);
    fillNormal(dp, generator, 1.0);
    const Scalar direction_norm = std::sqrt(
        du.squaredNorm() + dv.squaredNorm() + dp.squaredNorm());
    du /= direction_norm;
    dv /= direction_norm;
    dp /= direction_norm;

    const Scalar component_weights[][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.7, 1.3, 0.9}};
    const char* labels[] = {"mass", "xmom", "ymom", "combined"};
    const Scalar steps[] = {1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6};
    for (int component = 0; component < 4; ++component)
    {
        const Scalar wm = component_weights[component][0];
        const Scalar wx = component_weights[component][1];
        const Scalar wy = component_weights[component][2];
        const Evaluation base = evaluate(
            gl4.fit, u, v, p, wm, wx, wy, true);
        const Scalar derivative = gradientDot(base, du, dv, dp);
        Scalar minimum_error = std::numeric_limits<Scalar>::infinity();
        std::vector<Scalar> remainders;
        for (Scalar step : steps)
        {
            const Scalar plus = evaluate(
                gl4.fit,
                u + step * du, v + step * dv, p + step * dp,
                wm, wx, wy).weighted_loss;
            const Scalar minus = evaluate(
                gl4.fit,
                u - step * du, v - step * dv, p - step * dp,
                wm, wx, wy).weighted_loss;
            const Scalar finite_difference = (plus - minus) / (2.0 * step);
            const Scalar relative_error =
                std::abs(finite_difference - derivative) /
                std::max<Scalar>(1.0, std::max(
                    std::abs(finite_difference), std::abs(derivative)));
            minimum_error = std::min(minimum_error, relative_error);
            remainders.push_back(std::abs(
                plus - base.weighted_loss - step * derivative));
        }
        bool second_order = false;
        for (std::size_t index = 1; index + 1 < remainders.size(); ++index)
        {
            if (remainders[index] > 1.0e-15 && remainders[index - 1] > 0.0)
            {
                const Scalar slope = std::log(
                    remainders[index - 1] / remainders[index]) /
                    std::log(steps[index - 1] / steps[index]);
                if (std::isfinite(slope) && slope > 1.75 && slope < 2.25)
                    second_order = true;
            }
        }
        require(
            minimum_error < 2.0e-8,
            std::string("coefficient adjoint failed for ") + labels[component]);
        require(
            second_order,
            std::string("Taylor test failed for ") + labels[component]);
        std::cout
            << "NS_GL_ADJOINT component=" << labels[component]
            << " min_relative_error=" << minimum_error
            << " taylor_order=2 status=pass\n";
    }
}

void testZeroWeightRegression(const Fixture& fixture, const Operators& gl4)
{
    std::mt19937_64 generator(741u);
    Vector u(fixture.count()), v(fixture.count()), p(fixture.count());
    fillNormal(u, generator, 0.2);
    fillNormal(v, generator, 0.2);
    fillNormal(p, generator, 0.2);
    const Evaluation diagnostic = evaluate(
        gl4.validation, u, v, p, 0.0, 0.0, 0.0, true);
    require(
        diagnostic.mse_mass > 0.0 ||
        diagnostic.mse_xmom > 0.0 ||
        diagnostic.mse_ymom > 0.0,
        "zero-weight diagnostic did not evaluate residuals");
    require(diagnostic.weighted_loss == 0.0, "zero weights changed loss");
    require(
        diagnostic.coefficient_adjoints.u.isZero(0.0) &&
        diagnostic.coefficient_adjoints.v.isZero(0.0) &&
        diagnostic.coefficient_adjoints.p.isZero(0.0),
        "zero weights produced a coefficient adjoint");
    std::cout
        << "NS_GL_ZERO_WEIGHT status=pass"
        << " objective=0 adjoint=0 diagnostics=retained\n";
}

} // anonymous namespace

int main()
{
    try
    {
        const Fixture fixture;
        const Operators gl4 = Balance::build(
            fixture.knots, fixture.degrees, fixture.lower, fixture.upper,
            fixture.times, fixture.config, 4);
        const Operators gl8 = Balance::build(
            fixture.knots, fixture.degrees, fixture.lower, fixture.upper,
            fixture.times, fixture.config, 8);
        testManufacturedIdentity(fixture, gl4);
        testGLAgreementAndHierarchy(fixture, gl4, gl8);
        testAdjointAndTaylor(fixture, gl4);
        testZeroWeightRegression(fixture, gl4);
        std::cout
            << "NS_GL_SUMMARY status=pass"
            << " fit_rows=" << gl4.fit.rows()
            << " validation_rows=" << gl4.validation.rows()
            << " fit_face_samples=" << gl4.fit.faceSampleCount()
            << " quadrature=GL4_training_GL8_check\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "NS_GL_SUMMARY status=fail error="
                  << error.what() << "\n";
        return 1;
    }
}
