#include <mfa/burgers_knot_span_integral_balance.hpp>

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
typedef mfa::BurgersIntegralBalanceConfig<Scalar> Config;
typedef mfa::BurgersKnotSpanGLBalance<Scalar> Balance;
typedef mfa::BurgersGLBalanceOperators<Scalar> Operators;
typedef mfa::BurgersGLBlockOperators<Scalar> BlockOperators;
typedef mfa::BurgersGLBalanceEvaluation<Scalar> Evaluation;

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
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

std::vector<Scalar> openUniformKnots(int control_count)
{
    const int degree = 3;
    std::vector<Scalar> knots(control_count + degree + 1, 0.0);
    for (int index = control_count;
         index < static_cast<int>(knots.size()); ++index)
        knots[index] = 1.0;
    const int spans = control_count - degree;
    for (int interior = 1; interior < spans; ++interior)
        knots[degree + interior] =
            static_cast<Scalar>(interior) / static_cast<Scalar>(spans);
    return knots;
}

Scalar greville(const std::vector<Scalar>& knots, int control)
{
    return (knots[control + 1] + knots[control + 2] +
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
        knots[0] = openUniformKnots(8);
        knots[1] = openUniformKnots(7);
        knots[2] = openUniformKnots(6);
        degrees = {{3, 3, 3}};
        lower = {{-0.5, 0.25, 0.0}};
        upper = {{1.5, 1.75, 2.0}};
        times = {0.0, 0.2, 0.55, 1.1, 1.6, 2.0};
        config.lambda_u = 1.0;
        config.lambda_v = 1.0;
        config.parts_x = 4;
        config.parts_y = 4;
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
    Scalar area() const
    {
        return (upper[0] - lower[0]) * (upper[1] - lower[1]);
    }
};

void fillNormal(Vector& values, std::mt19937_64& generator, Scalar scale)
{
    std::normal_distribution<Scalar> distribution(0.0, scale);
    for (int index = 0; index < values.size(); ++index)
        values(index) = distribution(generator);
}

Evaluation evaluate(
    const BlockOperators& operators,
    const Vector& u,
    const Vector& v,
    Scalar weight_u,
    Scalar weight_v,
    bool adjoints = false)
{
    return Balance::evaluate(
        operators, u, v, 0.03, weight_u, weight_v, adjoints);
}

void testManufacturedFields(const Fixture& fixture, const Operators& gl4)
{
    Vector u = Vector::Zero(fixture.count());
    Vector v = Vector::Zero(fixture.count());
    Evaluation result = evaluate(gl4.fit, u, v, 1.0, 1.0);
    require(result.raw_u.isZero(0.0), "zero-field u balance is not exact zero");
    require(result.raw_v.isZero(0.0), "zero-field v balance is not exact zero");

    u.setConstant(0.7);
    v.setConstant(-0.2);
    result = evaluate(gl4.fit, u, v, 1.0, 1.0);
    require(
        result.raw_u.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "constant-field u balance failed");
    require(
        result.raw_v.lpNorm<Eigen::Infinity>() < 2.0e-13,
        "constant-field v balance failed");

    // Affine-in-time, spatially uniform fields have only storage residual.
    // Greville values reproduce affine functions exactly for a B-spline.
    const Scalar slope_u = 0.35;
    const Scalar slope_v = -0.17;
    for (int kt = 0; kt < fixture.nt(); ++kt)
        for (int ky = 0; ky < fixture.ny(); ++ky)
            for (int kx = 0; kx < fixture.nx(); ++kx)
            {
                const Scalar physical_t = fixture.lower[2] +
                    (fixture.upper[2] - fixture.lower[2]) *
                    greville(fixture.knots[2], kt);
                const int index = fixture.index(kx, ky, kt);
                u(index) = 0.4 + slope_u * physical_t;
                v(index) = -0.1 + slope_v * physical_t;
            }
    result = evaluate(gl4.fit, u, v, 1.0, 1.0);
    const Scalar block_area = fixture.area() /
        static_cast<Scalar>(fixture.config.parts_x * fixture.config.parts_y);
    for (int row = 0; row < result.raw_u.size(); ++row)
    {
        requireNear(
            result.raw_u(row), slope_u * block_area, 3.0e-12,
            "affine-time u storage failed");
        requireNear(
            result.raw_v(row), slope_v * block_area, 3.0e-12,
            "affine-time v storage failed");
    }
    require(
        gl4.fit.maximum_internal_face_closure == 0.0 &&
        gl4.validation.maximum_internal_face_closure == 0.0,
        "internal faces are not algebraically closed");
    std::cout
        << "BURGERS_GL_MANUFACTURED status=pass"
        << " cases=zero,constant,affine_time"
        << " internal_face_closure=exact\n";
}

void testFluxStrongEquivalenceAndGLAgreement(
    const Fixture& fixture,
    const Operators& gl4,
    const Operators& gl8)
{
    std::mt19937_64 generator(20260819u);
    Vector u(fixture.count()), v(fixture.count());
    fillNormal(u, generator, 0.25);
    fillNormal(v, generator, 0.20);
    const Evaluation four = evaluate(gl4.fit, u, v, 1.0, 1.0);
    const Evaluation eight = evaluate(gl8.fit, u, v, 1.0, 1.0);
    require(
        four.maximum_flux_strong_mismatch < 2.0e-11,
        "GL4 flux-plus-correction does not match direct strong integral");
    require(
        eight.maximum_flux_strong_mismatch < 2.0e-11,
        "GL8 flux-plus-correction does not match direct strong integral");
    const Scalar agreement = std::max(
        (four.raw_u - eight.raw_u).lpNorm<Eigen::Infinity>(),
        (four.raw_v - eight.raw_v).lpNorm<Eigen::Infinity>());
    require(agreement < 2.0e-11, "GL4/GL8 Burgers balance disagreement");
    std::cout
        << "BURGERS_GL_FORM_EQUIVALENCE status=pass"
        << " gl4_mismatch=" << four.maximum_flux_strong_mismatch
        << " gl8_mismatch=" << eight.maximum_flux_strong_mismatch
        << " gl4_gl8=" << agreement << "\n";
}

void testHierarchy(
    const Fixture& fixture,
    const Operators& q4,
    const Vector& u,
    const Vector& v)
{
    Config q2_config = fixture.config;
    q2_config.parts_x = 2;
    q2_config.parts_y = 2;
    Config q1_config = fixture.config;
    q1_config.parts_x = 1;
    q1_config.parts_y = 1;
    const Operators q2 = Balance::build(
        fixture.knots, fixture.degrees, fixture.lower, fixture.upper,
        fixture.times, q2_config, 4);
    const Operators q1 = Balance::build(
        fixture.knots, fixture.degrees, fixture.lower, fixture.upper,
        fixture.times, q1_config, 4);
    const Evaluation fine = evaluate(q4.fit, u, v, 1.0, 1.0);
    const Evaluation middle = evaluate(q2.fit, u, v, 1.0, 1.0);
    const Evaluation whole = evaluate(q1.fit, u, v, 1.0, 1.0);
    const Vector aggregate_u_q2 = Balance::aggregateRaw(
        q4.fit, fine.raw_u, 2, 2);
    const Vector aggregate_v_q2 = Balance::aggregateRaw(
        q4.fit, fine.raw_v, 2, 2);
    const Vector aggregate_u_q1 = Balance::aggregateRaw(
        q4.fit, fine.raw_u, 1, 1);
    const Vector aggregate_v_q1 = Balance::aggregateRaw(
        q4.fit, fine.raw_v, 1, 1);
    const Scalar mismatch = std::max({
        (aggregate_u_q2 - middle.raw_u).lpNorm<Eigen::Infinity>(),
        (aggregate_v_q2 - middle.raw_v).lpNorm<Eigen::Infinity>(),
        (aggregate_u_q1 - whole.raw_u).lpNorm<Eigen::Infinity>(),
        (aggregate_v_q1 - whole.raw_v).lpNorm<Eigen::Infinity>()});
    require(mismatch < 3.0e-11, "q4-to-q2/q1 hierarchy closure failed");
    std::cout
        << "BURGERS_GL_HIERARCHY status=pass"
        << " hierarchy=q4_to_q2_q1 max_mismatch=" << mismatch << "\n";
}

void testAdjoint(const Fixture& fixture, const Operators& gl4)
{
    std::mt19937_64 generator(991827u);
    Vector u(fixture.count()), v(fixture.count());
    Vector du(fixture.count()), dv(fixture.count());
    fillNormal(u, generator, 0.2);
    fillNormal(v, generator, 0.2);
    fillNormal(du, generator, 1.0);
    fillNormal(dv, generator, 1.0);
    const Scalar norm = std::sqrt(du.squaredNorm() + dv.squaredNorm());
    du /= norm;
    dv /= norm;
    const Scalar weights[][2] = {
        {1.0, 0.0}, {0.0, 1.0}, {0.7, 1.3}};
    const char* labels[] = {"u", "v", "combined"};
    const Scalar steps[] = {1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6};
    for (int component = 0; component < 3; ++component)
    {
        const Evaluation base = evaluate(
            gl4.fit, u, v, weights[component][0], weights[component][1], true);
        const Scalar analytic =
            base.coefficient_adjoints.u.dot(du) +
            base.coefficient_adjoints.v.dot(dv);
        Scalar minimum_error = std::numeric_limits<Scalar>::infinity();
        bool second_order = false;
        Scalar previous_remainder = -1.0;
        Scalar previous_step = -1.0;
        for (Scalar step : steps)
        {
            const Scalar plus = evaluate(
                gl4.fit, u + step * du, v + step * dv,
                weights[component][0], weights[component][1]).weighted_loss;
            const Scalar minus = evaluate(
                gl4.fit, u - step * du, v - step * dv,
                weights[component][0], weights[component][1]).weighted_loss;
            const Scalar finite_difference = (plus - minus) / (2.0 * step);
            const Scalar scale = std::max<Scalar>(
                1.0, std::max(std::abs(analytic), std::abs(finite_difference)));
            minimum_error = std::min(
                minimum_error, std::abs(analytic - finite_difference) / scale);
            const Scalar remainder = std::abs(
                plus - base.weighted_loss - step * analytic);
            if (previous_remainder > 1.0e-15 && remainder > 1.0e-15)
            {
                const Scalar slope = std::log(previous_remainder / remainder) /
                    std::log(previous_step / step);
                if (std::isfinite(slope) && slope > 1.75 && slope < 2.25)
                    second_order = true;
            }
            previous_remainder = remainder;
            previous_step = step;
        }
        require(
            minimum_error < 3.0e-8,
            std::string("coefficient adjoint failed for ") + labels[component]);
        require(
            second_order,
            std::string("Taylor remainder failed for ") + labels[component]);
        std::cout
            << "BURGERS_GL_ADJOINT component=" << labels[component]
            << " min_relative_error=" << minimum_error
            << " taylor_order=2 status=pass\n";
    }
}

void testZeroWeightRegression(
    const Fixture& fixture,
    const Operators& gl4)
{
    const Config disabled;
    require(
        !disabled.enabled() && !disabled.operatorsRequired(),
        "default Burgers integral config is not a strict disabled path");
    std::mt19937_64 generator(741u);
    Vector u(fixture.count()), v(fixture.count());
    fillNormal(u, generator, 0.2);
    fillNormal(v, generator, 0.2);
    const Evaluation diagnostic = evaluate(
        gl4.validation, u, v, 0.0, 0.0, true);
    require(
        diagnostic.mse_u > 0.0 || diagnostic.mse_v > 0.0,
        "zero-weight diagnostic did not evaluate residuals");
    require(diagnostic.weighted_loss == 0.0, "zero weights changed loss");
    require(
        diagnostic.coefficient_adjoints.u.isZero(0.0) &&
        diagnostic.coefficient_adjoints.v.isZero(0.0),
        "zero weights produced a coefficient adjoint");
    std::cout
        << "BURGERS_GL_ZERO_WEIGHT status=pass"
        << " objective=0 adjoint=0 default_fast_path=disabled\n";
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
        testManufacturedFields(fixture, gl4);
        testFluxStrongEquivalenceAndGLAgreement(fixture, gl4, gl8);
        std::mt19937_64 generator(8821u);
        Vector u(fixture.count()), v(fixture.count());
        fillNormal(u, generator, 0.2);
        fillNormal(v, generator, 0.2);
        testHierarchy(fixture, gl4, u, v);
        testAdjoint(fixture, gl4);
        testZeroWeightRegression(fixture, gl4);
        std::cout
            << "BURGERS_GL_SUMMARY status=pass"
            << " fit_rows=" << gl4.fit.rows()
            << " validation_rows=" << gl4.validation.rows()
            << " fit_face_samples=" << gl4.fit.faceSampleCount()
            << " fit_volume_samples=" << gl4.fit.volumeSampleCount()
            << " quadrature=GL4_training_GL8_check\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "BURGERS_GL_SUMMARY status=fail error="
                  << error.what() << "\n";
        return 1;
    }
}
