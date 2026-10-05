#include <mfa/ns_knot_span_integral_balance.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef mfa::NSKnotSpanGLBalance<Scalar> Balance;
typedef mfa::NSKnotSpanGLBalanceOperators<Scalar> Operators;
typedef mfa::NSKnotSpanGLBalanceEvaluation<Scalar> Evaluation;

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

std::vector<Scalar> readTimes(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open physical-time file: " + path);
    std::vector<Scalar> times;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream row(line);
        Scalar value = 0.0;
        if (!(row >> value) || !std::isfinite(value))
            throw std::runtime_error(
                "post-fit GL8 physical-time row is invalid or nonfinite");
        row >> std::ws;
        if (!row.eof())
            throw std::runtime_error(
                "post-fit GL8 physical-time row has trailing content");
        if (value < 0.0 || value > 10.0)
            throw std::runtime_error(
                "post-fit GL8 physical time is outside [0,10]");
        if (!times.empty() && !(value > times.back()))
            throw std::runtime_error(
                "post-fit GL8 physical times must be strictly increasing");
        times.push_back(value);
    }
    if (!input.eof() || times.empty())
        throw std::runtime_error(
            "post-fit GL8 physical-time file must be nonempty");
    return times;
}

Vector readCoefficients(const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
        throw std::runtime_error("cannot open coefficient file: " + path);
    std::string header;
    std::getline(input, header);
    if (header.find("n=60000") == std::string::npos)
        throw std::runtime_error(
            "post-fit GL8 coefficient header must contain n=60000");
    Vector coefficients(60000);
    for (int index = 0; index < coefficients.size(); ++index)
    {
        if (!(input >> coefficients(index)) ||
            !std::isfinite(coefficients(index)))
            throw std::runtime_error(
                "post-fit GL8 coefficient payload is incomplete or nonfinite");
    }
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error(
            "post-fit GL8 coefficient payload contains trailing content");
    return coefficients;
}

int parsePartition(const char* text)
{
    std::istringstream input(text);
    int value = 0;
    char trailing = '\0';
    if (!(input >> value) || (input >> trailing) || value < 1)
        throw std::invalid_argument(
            "post-fit GL8 partition q must be a positive integer");
    return value;
}

Operators buildOperators(const std::vector<Scalar>& times, int partition_q)
{
    if (partition_q < 1)
        throw std::invalid_argument("post-fit GL8 partition q must be positive");
    const std::array<std::vector<Scalar>, 3> knots = {{
        openUniformKnots(20),
        openUniformKnots(20),
        openUniformKnots(50)}};
    const std::array<int, 3> degrees = {{3, 3, 3}};
    const std::array<Scalar, 3> lower = {{0.0, 0.0, 0.0}};
    const std::array<Scalar, 3> upper = {{1.0, 1.0, 10.0}};
    std::vector<int> indices(times.size());
    std::iota(indices.begin(), indices.end(), 0);

    mfa::NSIntegralBalanceConfig<Scalar> config;
    config.diagnostics = true;
    config.parts_x = partition_q;
    config.parts_y = partition_q;
    config.physics_t_min = 0.0;
    config.viscosity = 0.001;
    config.check_closure = true;
    config.closure_tolerance = 1.0e-13;
    config.fit_time_indices = indices;
    // This executable evaluates one explicit post-fit time list.  The balance
    // builder also requires a validation role, which is not consumed here;
    // reusing the first explicit time avoids silently introducing another
    // physical time into the sealed calculation.
    config.validation_time_indices.push_back(indices.front());
    return Balance::build(
        knots, degrees, lower, upper, times, config, 8);
}

Evaluation evaluate(
    const Operators& operators,
    const Vector& all_coefficients)
{
    const Vector u = all_coefficients.segment(0, 20000);
    const Vector v = all_coefficients.segment(20000, 20000);
    const Vector p = all_coefficients.segment(40000, 20000);
    return Balance::evaluate(
        operators.fit, u, v, p, 0.001,
        0.0, 0.0, 0.0, false);
}

int selfTest()
{
    const std::vector<Scalar> times = {0.005, 4.995, 9.995};
    const Operators operators = buildOperators(times, 4);
    Vector coefficients(60000);
    coefficients.segment(0, 20000).setConstant(0.7);
    coefficients.segment(20000, 20000).setConstant(-0.2);
    coefficients.segment(40000, 20000).setConstant(4.0);
    const Evaluation result = evaluate(operators, coefficients);
    const Scalar maximum = std::max({
        result.raw_mass.lpNorm<Eigen::Infinity>(),
        result.raw_xmom.lpNorm<Eigen::Infinity>(),
        result.raw_ymom.lpNorm<Eigen::Infinity>()});
    if (!(maximum < 3.0e-12))
        throw std::runtime_error("post-fit GL8 constant-field self-test failed");
    std::cout
        << "NS_GL8_POSTFIT_SELF_TEST status=pass"
        << " max_residual=" << maximum
        << " quadrature=GL8 role=independent_postfit\n";
    return 0;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--self-test")
            return selfTest();
        if (argc != 4 && argc != 5)
            throw std::invalid_argument(
                "usage: ns-knot-span-postfit-gl8 COEFFICIENTS.txt "
                "PHYSICAL_TIMES.txt OUTPUT.tsv [PARTITION_Q]");
        const int partition_q = argc == 5 ? parsePartition(argv[4]) : 4;
        const Vector coefficients = readCoefficients(argv[1]);
        const std::vector<Scalar> times = readTimes(argv[2]);
        const Operators operators = buildOperators(times, partition_q);
        const Evaluation result = evaluate(operators, coefficients);

        std::ofstream output(argv[3]);
        if (!output)
            throw std::runtime_error(
                std::string("cannot create post-fit GL8 output: ") + argv[3]);
        output << std::scientific << std::setprecision(17);
        output
            << "time_index\ttime\tblock_x\tblock_y\tmass"
            << "\tx_momentum\ty_momentum\n";
        const int blocks = partition_q * partition_q;
        for (std::size_t time_position = 0;
             time_position < times.size(); ++time_position)
            for (int block_y = 0; block_y < partition_q; ++block_y)
                for (int block_x = 0; block_x < partition_q; ++block_x)
                {
                    const int row =
                        static_cast<int>(time_position) * blocks +
                        block_y * partition_q + block_x;
                    output
                        << time_position << '\t'
                        << times[time_position] << '\t'
                        << block_x << '\t' << block_y << '\t'
                        << result.raw_mass(row) << '\t'
                        << result.raw_xmom(row) << '\t'
                        << result.raw_ymom(row) << '\n';
                }
        std::cout
            << std::scientific << std::setprecision(17)
            << "NS_GL8_POSTFIT status=pass"
            << " quadrature=GL8 role=independent_postfit"
            << " partition_q=" << partition_q
            << " time_count=" << times.size()
            << " rows=" << result.raw_mass.size()
            << " mse_mass=" << result.mse_mass
            << " mse_xmom=" << result.mse_xmom
            << " mse_ymom=" << result.mse_ymom
            << " output=" << argv[3] << "\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "NS_GL8_POSTFIT status=fail error="
                  << error.what() << "\n";
        return 2;
    }
}
