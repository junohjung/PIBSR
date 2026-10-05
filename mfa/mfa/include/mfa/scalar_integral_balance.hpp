//--------------------------------------------------------------
// Representation-consistent finite-interval balances for the
// one-dimensional scalar convection--diffusion objective.
//
// For rho_t + a rho_x - nu rho_xx = 0 on K = [x_L,x_R],
//
//   b_K(t) = int_K rho_t dx
//          + [a rho - nu rho_x]_(x_R)
//          - [a rho - nu rho_x]_(x_L).
//
// Storage is integrated independently on every intersected knot span by
// Gauss--Legendre quadrature.  Each face is sampled once and referenced with
// opposite signs by neighboring intervals, so internal-face cancellation is
// algebraic.  The residual and its transpose act directly on the spline
// coefficients; no observation-grid quadrature or dense Jacobian is used.
//--------------------------------------------------------------

#ifndef MFA_SCALAR_INTEGRAL_BALANCE_HPP
#define MFA_SCALAR_INTEGRAL_BALANCE_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfa
{

template <typename T>
inline T scalarIntegralStrictScalar(
    const char* name,
    T default_value,
    bool nonnegative = true)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        return default_value;
    std::istringstream stream(text);
    T value;
    char trailing = '\0';
    if (!(stream >> value) || (stream >> trailing) || !std::isfinite(value))
        throw std::invalid_argument(
            std::string("Malformed finite scalar in ") + name + ": " + text);
    if (nonnegative && value < T(0))
        throw std::invalid_argument(std::string(name) + " must be nonnegative");
    return value;
}

inline int scalarIntegralStrictInteger(
    const char* name,
    int default_value)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        return default_value;
    std::istringstream stream(text);
    long long value;
    char trailing = '\0';
    if (!(stream >> value) || (stream >> trailing) ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            std::string("Malformed integer in ") + name + ": " + text);
    return static_cast<int>(value);
}

inline bool scalarIntegralStrictBoolean(const char* name, bool default_value)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        return default_value;
    if (std::string(text) == "0")
        return false;
    if (std::string(text) == "1")
        return true;
    throw std::invalid_argument(std::string(name) + " must be exactly 0 or 1");
}

inline int scalarIntegralQuadratureOrder()
{
    const char* text = std::getenv("MFA_INT_QUADRATURE");
    if (!text || !*text || std::string(text) == "gl4")
        return 4;
    if (std::string(text) == "gl8")
        return 8;
    throw std::invalid_argument(
        "MFA_INT_QUADRATURE must be exactly gl4 or gl8");
}

inline std::vector<int> scalarIntegralReadIndices(
    const char* environment_name,
    const char* label)
{
    const char* path = std::getenv(environment_name);
    if (!path || !*path)
        throw std::invalid_argument(
            std::string(environment_name) +
            " is required when the scalar integral operator is active");
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error(std::string("Cannot open ") + label + ": " + path);

    std::vector<int> indices;
    std::set<int> seen;
    long long value = -1;
    while (stream >> value)
    {
        if (value < 0 || value > std::numeric_limits<int>::max())
            throw std::runtime_error(std::string("Out-of-range index in ") + label);
        const int index = static_cast<int>(value);
        if (!seen.insert(index).second)
            throw std::runtime_error(std::string("Duplicate index in ") + label);
        indices.push_back(index);
    }
    if (!stream.eof() || indices.empty() ||
        !std::is_sorted(indices.begin(), indices.end()))
        throw std::runtime_error(
            std::string(label) + " must contain nonempty sorted integer indices");
    return indices;
}

template <typename T>
struct ScalarIntegralBalanceConfig
{
    T lambda_balance;
    bool diagnostics;
    int parts_x;
    int quadrature_order;
    bool check_closure;
    T closure_tolerance;
    std::vector<int> fit_time_indices;
    std::vector<int> validation_time_indices;

    ScalarIntegralBalanceConfig()
        : lambda_balance(T(0)),
          diagnostics(false),
          parts_x(1),
          quadrature_order(4),
          check_closure(true),
          closure_tolerance(T(1.0e-12))
    {
    }

    bool enabled() const { return lambda_balance > T(0); }
    bool operatorsRequired() const { return enabled() || diagnostics; }

    static ScalarIntegralBalanceConfig fromEnvironment()
    {
        ScalarIntegralBalanceConfig config;
        config.lambda_balance = scalarIntegralStrictScalar<T>(
            "MFA_LAMBDA_INT_BALANCE", T(0));
        config.diagnostics = scalarIntegralStrictBoolean(
            "MFA_INT_DIAGNOSTICS", false);

        // This early return is intentional: a baseline lambda=0 run does not
        // parse role files, allocate operators, or alter objective arithmetic.
        if (!config.operatorsRequired())
            return config;

        config.parts_x = scalarIntegralStrictInteger("MFA_INT_PARTS_X", 1);
        if (config.parts_x != 1 && config.parts_x != 4)
            throw std::invalid_argument(
                "MFA_INT_PARTS_X must be exactly 1 or 4");
        config.quadrature_order = scalarIntegralQuadratureOrder();
        config.fit_time_indices = scalarIntegralReadIndices(
            "MFA_INT_FIT_TIME_INDEX_FILE", "integral fit-time indices");
        config.validation_time_indices = scalarIntegralReadIndices(
            "MFA_INT_VALID_TIME_INDEX_FILE", "integral validation-time indices");
        validateDisjoint(config.fit_time_indices, config.validation_time_indices);
        return config;
    }

    static void validateDisjoint(
        const std::vector<int>& fit,
        const std::vector<int>& validation)
    {
        std::set<int> fit_set(fit.begin(), fit.end());
        for (std::size_t index = 0; index < validation.size(); ++index)
            if (fit_set.count(validation[index]))
                throw std::invalid_argument(
                    "Integral fit and validation time indices overlap");
    }
};

template <typename T>
struct ScalarIntegralBasis1D
{
    int first_control;
    T parameter;
    std::array<T, 4> value;
    std::array<T, 4> derivative;

    ScalarIntegralBasis1D() : first_control(0), parameter(T(0))
    {
        value.fill(T(0));
        derivative.fill(T(0));
    }
};

template <typename T>
struct ScalarIntegralIntegratedBasis1D
{
    std::vector<int> controls;
    std::vector<T> weights;
};

template <typename T>
struct ScalarIntegralRows
{
    typedef Eigen::SparseMatrix<T> SparseMatrix;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;

    // storage maps coefficients to int_K rho_t dx.  The two face matrices
    // map coefficients to rho and physical rho_x at unique face samples.
    SparseMatrix storage;
    SparseMatrix face_incidence;
    SparseMatrix face_value;
    SparseMatrix face_derivative_x;
    Vector scale;
    std::vector<int> time_indices;
    std::array<int, 2> control_counts;
    std::array<T, 2> physical_extents;
    int parts_x;
    int quadrature_order;
    T maximum_internal_face_closure;

    ScalarIntegralRows()
        : parts_x(0),
          quadrature_order(0),
          maximum_internal_face_closure(T(0))
    {
        control_counts.fill(0);
        physical_extents.fill(T(0));
    }

    int rows() const { return static_cast<int>(storage.rows()); }
    int coefficientCount() const
    {
        return control_counts[0] * control_counts[1];
    }
    int faceSampleCount() const
    {
        return static_cast<int>(face_value.rows());
    }
};

template <typename T>
struct ScalarIntegralBalanceOperators
{
    ScalarIntegralRows<T> fit;
    ScalarIntegralRows<T> validation;
    std::array<T, 2> physical_lower;
    std::array<T, 2> physical_upper;
    std::vector<T> x_boundaries;

    ScalarIntegralBalanceOperators()
    {
        physical_lower.fill(T(0));
        physical_upper.fill(T(0));
    }
};

template <typename T>
struct ScalarIntegralBalanceEvaluation
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector raw_balance;
    Vector scaled_balance;
    T mse;
    T weighted_loss;
    Vector coefficient_gradient;

    ScalarIntegralBalanceEvaluation() : mse(T(0)), weighted_loss(T(0)) {}
};

template <typename T>
class ScalarIntegralBalance
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Triplet<T> Triplet;

    // Knot values are parametric coordinates in [0,1].  Bounds and time_axis
    // are physical coordinates.  The implementation is deliberately limited
    // to the cubic, nonrational, affine tensor-product representation used by
    // the one-dimensional manuscript case.
    static ScalarIntegralBalanceOperators<T> build(
        const std::array<std::vector<T>, 2>& knots,
        const std::array<int, 2>& degrees,
        const std::array<T, 2>& physical_lower,
        const std::array<T, 2>& physical_upper,
        const std::vector<T>& time_axis,
        const ScalarIntegralBalanceConfig<T>& config,
        int quadrature_order = 4)
    {
        if (!config.operatorsRequired())
            throw std::logic_error(
                "ScalarIntegralBalance::build called while the operator is disabled");
        if (config.parts_x != 1 && config.parts_x != 4)
            throw std::invalid_argument(
                "Scalar integral partition count must be exactly 1 or 4");
        if (quadrature_order != 4 && quadrature_order != 8)
            throw std::invalid_argument(
                "Scalar knot-span quadrature order must be 4 or 8");
        if (!std::isfinite(config.closure_tolerance) ||
            config.closure_tolerance < T(0))
            throw std::invalid_argument(
                "Scalar integral closure tolerance must be finite and nonnegative");
        ScalarIntegralBalanceConfig<T>::validateDisjoint(
            config.fit_time_indices, config.validation_time_indices);

        for (int axis = 0; axis < 2; ++axis)
        {
            if (degrees[axis] != 3)
                throw std::invalid_argument(
                    "Scalar knot-span balance currently requires cubic splines");
            validateKnotVector(knots[axis], degrees[axis], axis);
            if (!std::isfinite(physical_lower[axis]) ||
                !std::isfinite(physical_upper[axis]) ||
                !(physical_upper[axis] > physical_lower[axis]))
                throw std::invalid_argument(
                    "Scalar knot-span physical bounds are invalid");
        }
        validateTimeAxis(time_axis, physical_lower[1], physical_upper[1]);
        validateTimes(
            config.fit_time_indices, time_axis,
            physical_lower[1], physical_upper[1], "fit");
        validateTimes(
            config.validation_time_indices, time_axis,
            physical_lower[1], physical_upper[1], "validation");

        ScalarIntegralBalanceOperators<T> result;
        result.physical_lower = physical_lower;
        result.physical_upper = physical_upper;
        result.fit = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.fit_time_indices, config.parts_x, quadrature_order,
            config.check_closure, config.closure_tolerance);
        result.validation = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.validation_time_indices, config.parts_x, quadrature_order,
            config.check_closure, config.closure_tolerance);
        for (int part = 0; part <= config.parts_x; ++part)
            result.x_boundaries.push_back(
                physical_lower[0] +
                (physical_upper[0] - physical_lower[0]) *
                static_cast<T>(part) / static_cast<T>(config.parts_x));
        return result;
    }

    static ScalarIntegralBalanceEvaluation<T> evaluate(
        const ScalarIntegralRows<T>& operators,
        const Vector& coefficients,
        T advection,
        T diffusivity,
        T lambda_balance,
        bool compute_gradient)
    {
        requireCoefficients(
            coefficients, operators.coefficientCount(), "rho");
        if (operators.rows() <= 0)
            throw std::invalid_argument(
                "Scalar knot-span balance requires nonempty operator rows");
        if (!std::isfinite(advection))
            throw std::invalid_argument(
                "Scalar knot-span advection must be finite");
        if (!(diffusivity > T(0)) || !std::isfinite(diffusivity))
            throw std::invalid_argument(
                "Scalar knot-span diffusivity must be positive and finite");
        if (lambda_balance < T(0) || !std::isfinite(lambda_balance))
            throw std::invalid_argument(
                "Scalar knot-span weight must be finite and nonnegative");

        ScalarIntegralBalanceEvaluation<T> result;
        const Vector face_flux =
            advection * (operators.face_value * coefficients) -
            diffusivity * (operators.face_derivative_x * coefficients);
        result.raw_balance =
            operators.storage * coefficients +
            operators.face_incidence * face_flux;
        result.scaled_balance =
            operators.scale.cwiseProduct(result.raw_balance);
        const T inverse_rows = T(1) / static_cast<T>(operators.rows());
        result.mse = result.scaled_balance.squaredNorm() * inverse_rows;
        result.weighted_loss = lambda_balance * result.mse;

        if (!compute_gradient)
            return result;
        result.coefficient_gradient =
            Vector::Zero(operators.coefficientCount());
        if (lambda_balance == T(0))
            return result;

        const Vector row_adjoint =
            (T(2) * lambda_balance * inverse_rows) *
            operators.scale.cwiseProduct(
                operators.scale.cwiseProduct(result.raw_balance));
        const Vector face_adjoint =
            operators.face_incidence.transpose() * row_adjoint;
        result.coefficient_gradient =
            operators.storage.transpose() * row_adjoint +
            advection * operators.face_value.transpose() * face_adjoint -
            diffusivity *
                operators.face_derivative_x.transpose() * face_adjoint;
        return result;
    }

private:
    struct GaussRule
    {
        std::vector<T> nodes;
        std::vector<T> weights;
    };

    static GaussRule gaussRule(int order)
    {
        GaussRule rule;
        if (order == 4)
        {
            const T nodes[] = {
                T(-0.86113631159405257522),
                T(-0.33998104358485626480),
                T( 0.33998104358485626480),
                T( 0.86113631159405257522)};
            const T weights[] = {
                T(0.34785484513745385737),
                T(0.65214515486254614263),
                T(0.65214515486254614263),
                T(0.34785484513745385737)};
            rule.nodes.assign(nodes, nodes + 4);
            rule.weights.assign(weights, weights + 4);
        }
        else if (order == 8)
        {
            const T nodes[] = {
                T(-0.96028985649753623168),
                T(-0.79666647741362673959),
                T(-0.52553240991632898582),
                T(-0.18343464249564980494),
                T( 0.18343464249564980494),
                T( 0.52553240991632898582),
                T( 0.79666647741362673959),
                T( 0.96028985649753623168)};
            const T weights[] = {
                T(0.10122853629037625915),
                T(0.22238103445337447054),
                T(0.31370664587788728734),
                T(0.36268378337836198297),
                T(0.36268378337836198297),
                T(0.31370664587788728734),
                T(0.22238103445337447054),
                T(0.10122853629037625915)};
            rule.nodes.assign(nodes, nodes + 8);
            rule.weights.assign(weights, weights + 8);
        }
        else
            throw std::invalid_argument(
                "Scalar knot-span quadrature order must be 4 or 8");
        return rule;
    }

    static void validateKnotVector(
        const std::vector<T>& knots,
        int degree,
        int axis)
    {
        if (knots.size() < static_cast<std::size_t>(2 * degree + 2))
            throw std::invalid_argument(
                "Scalar knot-span knot vector is too short");
        for (std::size_t index = 0; index < knots.size(); ++index)
        {
            if (!std::isfinite(knots[index]) ||
                knots[index] < T(0) || knots[index] > T(1) ||
                (index > 0 && knots[index] < knots[index - 1]))
                throw std::invalid_argument(
                    "Scalar knot-span knot vector is invalid on axis " +
                    std::to_string(axis));
        }
        for (int index = 0; index <= degree; ++index)
            if (knots[index] != T(0) ||
                knots[knots.size() - 1 - index] != T(1))
                throw std::invalid_argument(
                    "Scalar knot-span balance requires clamped [0,1] knots");
    }

    static void validateTimeAxis(
        const std::vector<T>& time_axis,
        T physical_lower,
        T physical_upper)
    {
        if (time_axis.empty())
            throw std::invalid_argument(
                "Scalar knot-span balance requires a nonempty time axis");
        const T tolerance = T(256) * std::numeric_limits<T>::epsilon() *
            std::max<T>(T(1), std::max(std::abs(physical_lower),
                                      std::abs(physical_upper)));
        for (std::size_t index = 0; index < time_axis.size(); ++index)
        {
            if (!std::isfinite(time_axis[index]) ||
                (index > 0 && !(time_axis[index] > time_axis[index - 1])) ||
                time_axis[index] < physical_lower - tolerance ||
                time_axis[index] > physical_upper + tolerance)
                throw std::invalid_argument(
                    "Scalar knot-span time axis must be finite, increasing, "
                    "and inside the physical time bounds");
        }
    }

    static void validateTimes(
        const std::vector<int>& indices,
        const std::vector<T>& time_axis,
        T physical_lower,
        T physical_upper,
        const char* label)
    {
        if (indices.empty())
            throw std::invalid_argument(
                std::string("Scalar knot-span ") + label +
                " time-index set is empty");
        const T scale = std::max<T>(
            T(1), std::max(std::abs(physical_lower), std::abs(physical_upper)));
        const T tolerance = T(256) * std::numeric_limits<T>::epsilon() * scale;
        for (std::size_t position = 0; position < indices.size(); ++position)
        {
            const int index = indices[position];
            if (index < 0 || index >= static_cast<int>(time_axis.size()))
                throw std::invalid_argument(
                    std::string("Scalar knot-span ") + label +
                    " time index is out of range");
            if (position > 0 && index <= indices[position - 1])
                throw std::invalid_argument(
                    std::string("Scalar knot-span ") + label +
                    " time indices must be strictly increasing");
            if (time_axis[index] <= physical_lower + tolerance)
                throw std::invalid_argument(
                    std::string("Scalar knot-span ") + label +
                    " roles must exclude the initial-time plane");
            if (time_axis[index] > physical_upper + tolerance)
                throw std::invalid_argument(
                    std::string("Scalar knot-span ") + label +
                    " time index is outside the physical domain");
        }
    }

    static int findSpan(T parameter, const std::vector<T>& knots)
    {
        const int control_count = static_cast<int>(knots.size()) - 4;
        if (parameter == T(1))
            return control_count - 1;
        int low = 3;
        int high = control_count;
        int middle = (low + high) / 2;
        while (parameter < knots[middle] ||
               parameter >= knots[middle + 1])
        {
            if (parameter < knots[middle])
                high = middle;
            else
                low = middle;
            middle = (low + high) / 2;
        }
        return middle;
    }

    static ScalarIntegralBasis1D<T> basis(
        T parameter,
        const std::vector<T>& knots)
    {
        if (!std::isfinite(parameter) ||
            parameter < T(0) || parameter > T(1))
            throw std::out_of_range(
                "Scalar knot-span basis parameter lies outside [0,1]");
        const int degree = 3;
        const int span = findSpan(parameter, knots);
        T ndu[4][4] = {};
        T left[4] = {};
        T right[4] = {};
        ndu[0][0] = T(1);
        for (int column = 1; column <= degree; ++column)
        {
            left[column] = parameter - knots[span + 1 - column];
            right[column] = knots[span + column] - parameter;
            T saved = T(0);
            for (int row = 0; row < column; ++row)
            {
                ndu[column][row] = right[row + 1] + left[column - row];
                if (ndu[column][row] == T(0))
                    throw std::runtime_error(
                        "Scalar knot-span basis encountered a zero denominator");
                const T temporary =
                    ndu[row][column - 1] / ndu[column][row];
                ndu[row][column] = saved + right[row + 1] * temporary;
                saved = left[column - row] * temporary;
            }
            ndu[column][column] = saved;
        }

        ScalarIntegralBasis1D<T> result;
        result.first_control = span - degree;
        result.parameter = parameter;
        for (int local = 0; local <= degree; ++local)
            result.value[local] = ndu[local][degree];

        T workspace[2][4] = {};
        for (int local = 0; local <= degree; ++local)
        {
            workspace[0][0] = T(1);
            T value = T(0);
            const int reduced_local = local - 1;
            const int reduced_degree = degree - 1;
            if (local >= 1)
            {
                const T denominator =
                    ndu[reduced_degree + 1][reduced_local];
                if (denominator == T(0))
                    throw std::runtime_error(
                        "Scalar knot-span derivative encountered a zero denominator");
                workspace[1][0] = workspace[0][0] / denominator;
                value = workspace[1][0] *
                    ndu[reduced_local][reduced_degree];
            }
            const int first = reduced_local >= -1 ? 1 : -reduced_local;
            const int last = local - 1 <= reduced_degree
                ? 0 : degree - local;
            for (int offset = first; offset <= last; ++offset)
            {
                const T denominator =
                    ndu[reduced_degree + 1][reduced_local + offset];
                if (denominator == T(0))
                    throw std::runtime_error(
                        "Scalar knot-span derivative encountered a zero denominator");
                workspace[1][offset] =
                    (workspace[0][offset] - workspace[0][offset - 1]) /
                    denominator;
                value += workspace[1][offset] *
                    ndu[reduced_local + offset][reduced_degree];
            }
            if (local <= reduced_degree)
            {
                const T denominator =
                    ndu[reduced_degree + 1][local];
                if (denominator == T(0))
                    throw std::runtime_error(
                        "Scalar knot-span derivative encountered a zero denominator");
                workspace[1][1] = -workspace[0][0] / denominator;
                value += workspace[1][1] * ndu[local][reduced_degree];
            }
            result.derivative[local] = static_cast<T>(degree) * value;
        }
        return result;
    }

    static std::vector<T> uniqueKnots(const std::vector<T>& knots)
    {
        std::vector<T> result = knots;
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    static std::vector<std::pair<T, T> > clippedSpans(
        T lower,
        T upper,
        const std::vector<T>& unique_knots)
    {
        std::vector<T> boundaries;
        boundaries.push_back(lower);
        for (std::size_t index = 0; index < unique_knots.size(); ++index)
            if (unique_knots[index] > lower && unique_knots[index] < upper)
                boundaries.push_back(unique_knots[index]);
        boundaries.push_back(upper);
        std::vector<std::pair<T, T> > result;
        for (std::size_t index = 0; index + 1 < boundaries.size(); ++index)
        {
            if (!(boundaries[index + 1] > boundaries[index]))
                throw std::runtime_error(
                    "Scalar knot-span balance produced a nonpositive span");
            result.push_back(std::make_pair(
                boundaries[index], boundaries[index + 1]));
        }
        return result;
    }

    static ScalarIntegralIntegratedBasis1D<T> integratedBasis(
        T lower,
        T upper,
        T physical_extent,
        const std::vector<T>& knots,
        int quadrature_order)
    {
        const int control_count = static_cast<int>(knots.size()) - 4;
        std::vector<T> dense(control_count, T(0));
        const GaussRule rule = gaussRule(quadrature_order);
        const std::vector<std::pair<T, T> > spans =
            clippedSpans(lower, upper, uniqueKnots(knots));
        for (std::size_t span_index = 0;
             span_index < spans.size(); ++span_index)
        {
            const T midpoint =
                T(0.5) * (spans[span_index].first + spans[span_index].second);
            const T half_width =
                T(0.5) * (spans[span_index].second - spans[span_index].first);
            for (std::size_t node = 0; node < rule.nodes.size(); ++node)
            {
                const T parameter = midpoint + half_width * rule.nodes[node];
                const T weight =
                    physical_extent * half_width * rule.weights[node];
                const ScalarIntegralBasis1D<T> local = basis(parameter, knots);
                for (int offset = 0; offset < 4; ++offset)
                    dense[local.first_control + offset] +=
                        weight * local.value[offset];
            }
        }
        ScalarIntegralIntegratedBasis1D<T> result;
        for (int control = 0; control < control_count; ++control)
            if (dense[control] != T(0))
            {
                result.controls.push_back(control);
                result.weights.push_back(dense[control]);
            }
        return result;
    }

    static int coefficientIndex(
        int x,
        int t,
        int control_count_x)
    {
        return x + control_count_x * t;
    }

    static ScalarIntegralRows<T> buildRows(
        const std::array<std::vector<T>, 2>& knots,
        const std::array<T, 2>& physical_lower,
        const std::array<T, 2>& physical_upper,
        const std::vector<T>& time_axis,
        const std::vector<int>& time_indices,
        int parts_x,
        int quadrature_order,
        bool check_closure,
        T closure_tolerance)
    {
        ScalarIntegralRows<T> result;
        result.parts_x = parts_x;
        result.quadrature_order = quadrature_order;
        result.time_indices = time_indices;
        for (int axis = 0; axis < 2; ++axis)
        {
            result.control_counts[axis] =
                static_cast<int>(knots[axis].size()) - 4;
            result.physical_extents[axis] =
                physical_upper[axis] - physical_lower[axis];
        }

        std::vector<ScalarIntegralIntegratedBasis1D<T> > integrated_x;
        for (int part = 0; part < parts_x; ++part)
            integrated_x.push_back(integratedBasis(
                static_cast<T>(part) / static_cast<T>(parts_x),
                static_cast<T>(part + 1) / static_cast<T>(parts_x),
                result.physical_extents[0], knots[0], quadrature_order));

        const int rows = parts_x * static_cast<int>(time_indices.size());
        const int face_count =
            (parts_x + 1) * static_cast<int>(time_indices.size());
        const int coefficient_count =
            result.control_counts[0] * result.control_counts[1];
        result.scale.resize(rows);
        result.storage.resize(rows, coefficient_count);
        result.face_incidence.resize(rows, face_count);
        result.face_value.resize(face_count, coefficient_count);
        result.face_derivative_x.resize(face_count, coefficient_count);

        std::vector<Triplet> storage_triplets;
        std::vector<Triplet> incidence_triplets;
        std::vector<Triplet> face_value_triplets;
        std::vector<Triplet> face_derivative_triplets;
        storage_triplets.reserve(
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(result.control_counts[0]) * 4);
        incidence_triplets.reserve(static_cast<std::size_t>(rows) * 2);
        face_value_triplets.reserve(static_cast<std::size_t>(face_count) * 16);
        face_derivative_triplets.reserve(
            static_cast<std::size_t>(face_count) * 16);

        for (std::size_t time_position = 0;
             time_position < time_indices.size(); ++time_position)
        {
            const T physical_time = time_axis[time_indices[time_position]];
            T parameter_time =
                (physical_time - physical_lower[1]) /
                result.physical_extents[1];
            const T parameter_tolerance =
                T(64) * std::numeric_limits<T>::epsilon();
            if (parameter_time < T(0) &&
                parameter_time > -parameter_tolerance)
                parameter_time = T(0);
            if (parameter_time > T(1) &&
                parameter_time < T(1) + parameter_tolerance)
                parameter_time = T(1);
            const ScalarIntegralBasis1D<T> time_basis =
                basis(parameter_time, knots[1]);

            for (int fixed = 0; fixed <= parts_x; ++fixed)
            {
                const int face =
                    static_cast<int>(time_position) * (parts_x + 1) + fixed;
                const T parameter_x =
                    static_cast<T>(fixed) / static_cast<T>(parts_x);
                const ScalarIntegralBasis1D<T> x_basis =
                    basis(parameter_x, knots[0]);
                for (int local_t = 0; local_t < 4; ++local_t)
                    for (int local_x = 0; local_x < 4; ++local_x)
                    {
                        const int coefficient = coefficientIndex(
                            x_basis.first_control + local_x,
                            time_basis.first_control + local_t,
                            result.control_counts[0]);
                        const T common = time_basis.value[local_t];
                        face_value_triplets.push_back(Triplet(
                            face, coefficient,
                            x_basis.value[local_x] * common));
                        face_derivative_triplets.push_back(Triplet(
                            face, coefficient,
                            x_basis.derivative[local_x] * common /
                                result.physical_extents[0]));
                    }
            }

            for (int part = 0; part < parts_x; ++part)
            {
                const int row =
                    static_cast<int>(time_position) * parts_x + part;
                const T interval_length =
                    result.physical_extents[0] / static_cast<T>(parts_x);
                result.scale(row) = T(1) / std::sqrt(interval_length);
                const int left_face =
                    static_cast<int>(time_position) * (parts_x + 1) + part;
                incidence_triplets.push_back(
                    Triplet(row, left_face, T(-1)));
                incidence_triplets.push_back(
                    Triplet(row, left_face + 1, T(1)));

                const ScalarIntegralIntegratedBasis1D<T>& x =
                    integrated_x[part];
                for (int local_t = 0; local_t < 4; ++local_t)
                {
                    const T dt = time_basis.derivative[local_t] /
                        result.physical_extents[1];
                    for (std::size_t local_x = 0;
                         local_x < x.controls.size(); ++local_x)
                    {
                        const int coefficient = coefficientIndex(
                            x.controls[local_x],
                            time_basis.first_control + local_t,
                            result.control_counts[0]);
                        storage_triplets.push_back(Triplet(
                            row, coefficient, dt * x.weights[local_x]));
                    }
                }
            }
        }

        result.storage.setFromTriplets(
            storage_triplets.begin(), storage_triplets.end());
        result.face_incidence.setFromTriplets(
            incidence_triplets.begin(), incidence_triplets.end());
        result.face_value.setFromTriplets(
            face_value_triplets.begin(), face_value_triplets.end());
        result.face_derivative_x.setFromTriplets(
            face_derivative_triplets.begin(),
            face_derivative_triplets.end());
        result.storage.makeCompressed();
        result.face_incidence.makeCompressed();
        result.face_value.makeCompressed();
        result.face_derivative_x.makeCompressed();

        const Vector ones = Vector::Ones(rows);
        const Vector incidence_sum =
            result.face_incidence.transpose() * ones;
        T maximum_internal = T(0);
        for (std::size_t time_position = 0;
             time_position < time_indices.size(); ++time_position)
            for (int fixed = 1; fixed < parts_x; ++fixed)
            {
                const int face =
                    static_cast<int>(time_position) * (parts_x + 1) + fixed;
                maximum_internal = std::max(
                    maximum_internal, std::abs(incidence_sum(face)));
            }
        result.maximum_internal_face_closure = maximum_internal;
        if (check_closure && maximum_internal > closure_tolerance)
            throw std::runtime_error(
                "Scalar knot-span internal-face closure check failed");
        return result;
    }

    static void requireCoefficients(
        const Vector& coefficients,
        int expected,
        const char* label)
    {
        if (coefficients.size() != expected || !coefficients.allFinite())
            throw std::invalid_argument(
                std::string("Invalid scalar knot-span coefficients for ") + label);
    }
};

} // namespace mfa

#endif // MFA_SCALAR_INTEGRAL_BALANCE_HPP
