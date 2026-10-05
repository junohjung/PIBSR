//--------------------------------------------------------------
// Representation-consistent finite-region balances for cubic,
// nonrational tensor-product B-splines.
//
// Spatial volume and face terms are integrated independently on every
// knot span with Gauss--Legendre quadrature.  A face is constructed once
// and referenced with opposite signs by its two neighboring regions, so
// internal-face cancellation is algebraic rather than tolerance-based.
// The residual Jacobian transpose is applied directly to the B-spline
// coefficients; no dense coefficient-space Jacobian is formed.
//--------------------------------------------------------------

#ifndef MFA_NS_KNOT_SPAN_INTEGRAL_BALANCE_HPP
#define MFA_NS_KNOT_SPAN_INTEGRAL_BALANCE_HPP

#include <mfa/ns_integral_balance.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfa
{

template <typename T>
struct NSKnotSpanGLBasis1D
{
    int first_control;
    T parameter;
    std::array<T, 4> value;
    std::array<T, 4> derivative;

    NSKnotSpanGLBasis1D() : first_control(0), parameter(T(0))
    {
        value.fill(T(0));
        derivative.fill(T(0));
    }
};

template <typename T>
struct NSKnotSpanGLIntegratedBasis1D
{
    std::vector<int> controls;
    std::vector<T> weights;
};

struct NSKnotSpanGLFaceSample
{
    int x_basis;
    int y_basis;
    int t_basis;
    int normal_axis;
    int fixed_partition_index;
};

template <typename T>
struct NSKnotSpanGLBlockOperators
{
    typedef Eigen::SparseMatrix<T> SparseMatrix;

    SparseMatrix Ex;
    SparseMatrix Ey;
    Eigen::Matrix<T, Eigen::Dynamic, 1> scale_mass;
    Eigen::Matrix<T, Eigen::Dynamic, 1> scale_momentum;
    std::vector<NSKnotSpanGLFaceSample> face_samples;
    std::vector<NSKnotSpanGLBasis1D<T> > x_bases;
    std::vector<NSKnotSpanGLBasis1D<T> > y_bases;
    std::vector<NSKnotSpanGLBasis1D<T> > t_bases;
    std::vector<int> time_basis_indices;
    std::vector<int> time_indices;
    std::vector<NSKnotSpanGLIntegratedBasis1D<T> > integrated_x;
    std::vector<NSKnotSpanGLIntegratedBasis1D<T> > integrated_y;
    std::array<int, 3> control_counts;
    std::array<T, 3> physical_extents;
    int parts_x;
    int parts_y;
    int quadrature_order;
    T maximum_internal_face_closure;

    NSKnotSpanGLBlockOperators()
        : parts_x(0),
          parts_y(0),
          quadrature_order(0),
          maximum_internal_face_closure(T(0))
    {
        control_counts.fill(0);
        physical_extents.fill(T(0));
    }

    int rows() const { return static_cast<int>(Ex.rows()); }
    int blocks() const { return parts_x * parts_y; }
    int coefficientCount() const
    {
        return control_counts[0] * control_counts[1] * control_counts[2];
    }
    std::size_t faceSampleCount() const { return face_samples.size(); }
};

template <typename T>
struct NSKnotSpanGLBalanceOperators
{
    NSKnotSpanGLBlockOperators<T> fit;
    NSKnotSpanGLBlockOperators<T> validation;
    std::array<T, 3> physical_lower;
    std::array<T, 3> physical_upper;
    std::vector<T> x_boundaries;
    std::vector<T> y_boundaries;
};

template <typename T>
struct NSKnotSpanGLCoefficientAdjoints
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector u;
    Vector v;
    Vector p;

    void resizeAndZero(int count)
    {
        u = Vector::Zero(count);
        v = Vector::Zero(count);
        p = Vector::Zero(count);
    }
};

template <typename T>
struct NSKnotSpanGLBalanceEvaluation
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector raw_mass;
    Vector raw_xmom;
    Vector raw_ymom;
    T mse_mass;
    T mse_xmom;
    T mse_ymom;
    T weighted_loss;
    NSKnotSpanGLCoefficientAdjoints<T> coefficient_adjoints;

    NSKnotSpanGLBalanceEvaluation()
        : mse_mass(T(0)),
          mse_xmom(T(0)),
          mse_ymom(T(0)),
          weighted_loss(T(0))
    {
    }
};

template <typename T>
class NSKnotSpanGLBalance
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Triplet<T> Triplet;

    // The implementation is deliberately restricted to the cubic,
    // nonrational tensor-product representation used by the manuscript.
    // The knot values are parametric coordinates in [0,1]; bounds and
    // time_axis are physical coordinates.
    static NSKnotSpanGLBalanceOperators<T> build(
        const std::array<std::vector<T>, 3>& knots,
        const std::array<int, 3>& degrees,
        const std::array<T, 3>& physical_lower,
        const std::array<T, 3>& physical_upper,
        const std::vector<T>& time_axis,
        const NSIntegralBalanceConfig<T>& config,
        int quadrature_order)
    {
        if (!config.operatorsRequired())
            throw std::logic_error(
                "NSKnotSpanGLBalance::build called without objective "
                "weights or diagnostic mode");
        if (quadrature_order != 4 && quadrature_order != 8)
            throw std::invalid_argument(
                "Knot-span quadrature order must be 4 or 8");
        for (int axis = 0; axis < 3; ++axis)
        {
            if (degrees[axis] != 3)
                throw std::invalid_argument(
                    "Knot-span balance currently requires cubic splines");
            validateKnotVector(knots[axis], degrees[axis], axis);
            if (!std::isfinite(physical_lower[axis]) ||
                !std::isfinite(physical_upper[axis]) ||
                !(physical_upper[axis] > physical_lower[axis]))
                throw std::invalid_argument(
                    "Knot-span balance physical bounds are invalid");
        }
        if (time_axis.empty())
            throw std::invalid_argument(
                "Knot-span balance requires a nonempty time axis");
        for (std::size_t index = 0; index < time_axis.size(); ++index)
        {
            if (!std::isfinite(time_axis[index]) ||
                (index > 0 && !(time_axis[index] > time_axis[index - 1])))
                throw std::invalid_argument(
                    "Knot-span balance time axis must be finite and "
                    "strictly increasing");
        }
        validateTimes(
            config.fit_time_indices, time_axis,
            physical_lower[2], physical_upper[2],
            config.physics_t_min, "fit");
        validateTimes(
            config.validation_time_indices, time_axis,
            physical_lower[2], physical_upper[2],
            config.physics_t_min, "validation");

        NSKnotSpanGLBalanceOperators<T> result;
        result.physical_lower = physical_lower;
        result.physical_upper = physical_upper;
        result.fit = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.fit_time_indices, config.parts_x, config.parts_y,
            quadrature_order, config.check_closure,
            config.closure_tolerance);
        result.validation = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.validation_time_indices,
            config.parts_x, config.parts_y,
            quadrature_order, config.check_closure,
            config.closure_tolerance);
        for (int part = 0; part <= config.parts_x; ++part)
            result.x_boundaries.push_back(
                physical_lower[0] +
                (physical_upper[0] - physical_lower[0]) *
                static_cast<T>(part) / static_cast<T>(config.parts_x));
        for (int part = 0; part <= config.parts_y; ++part)
            result.y_boundaries.push_back(
                physical_lower[1] +
                (physical_upper[1] - physical_lower[1]) *
                static_cast<T>(part) / static_cast<T>(config.parts_y));
        return result;
    }

    static NSKnotSpanGLBalanceEvaluation<T> evaluate(
        const NSKnotSpanGLBlockOperators<T>& operators,
        const Vector& coefficients_u,
        const Vector& coefficients_v,
        const Vector& coefficients_p,
        T viscosity,
        T lambda_mass,
        T lambda_xmom,
        T lambda_ymom,
        bool compute_adjoints)
    {
        const int coefficient_count = operators.coefficientCount();
        requireCoefficients(coefficients_u, coefficient_count, "u");
        requireCoefficients(coefficients_v, coefficient_count, "v");
        requireCoefficients(coefficients_p, coefficient_count, "p");
        if (operators.rows() <= 0)
            throw std::invalid_argument(
                "Knot-span balance requires nonempty operator rows");
        if (!(viscosity > T(0)) || !std::isfinite(viscosity))
            throw std::invalid_argument(
                "Knot-span balance viscosity must be positive and finite");
        if (lambda_mass < T(0) || lambda_xmom < T(0) ||
            lambda_ymom < T(0) || !std::isfinite(lambda_mass) ||
            !std::isfinite(lambda_xmom) || !std::isfinite(lambda_ymom))
            throw std::invalid_argument(
                "Knot-span balance weights must be finite and nonnegative");

        const int sample_count =
            static_cast<int>(operators.face_samples.size());
        Vector u(sample_count), ux(sample_count), uy(sample_count);
        Vector v(sample_count), vx(sample_count), vy(sample_count);
        Vector p(sample_count);
        for (int sample = 0; sample < sample_count; ++sample)
        {
            evaluateFaceSample(
                operators, operators.face_samples[sample], coefficients_u,
                u(sample), ux(sample), uy(sample));
            evaluateFaceSample(
                operators, operators.face_samples[sample], coefficients_v,
                v(sample), vx(sample), vy(sample));
            T unused_x = T(0);
            T unused_y = T(0);
            evaluateFaceSample(
                operators, operators.face_samples[sample], coefficients_p,
                p(sample), unused_x, unused_y);
        }

        NSKnotSpanGLBalanceEvaluation<T> result;
        const Vector storage_u = storageValues(operators, coefficients_u);
        const Vector storage_v = storageValues(operators, coefficients_v);
        const Vector flux_x_x =
            u.cwiseProduct(u) + p - viscosity * ux;
        const Vector flux_y_x =
            u.cwiseProduct(v) - viscosity * uy;
        const Vector flux_x_y =
            u.cwiseProduct(v) - viscosity * vx;
        const Vector flux_y_y =
            v.cwiseProduct(v) + p - viscosity * vy;

        result.raw_mass = operators.Ex * u + operators.Ey * v;
        result.raw_xmom =
            storage_u + operators.Ex * flux_x_x + operators.Ey * flux_y_x;
        result.raw_ymom =
            storage_v + operators.Ex * flux_x_y + operators.Ey * flux_y_y;

        const T inverse_rows = T(1) / static_cast<T>(operators.rows());
        result.mse_mass =
            (operators.scale_mass.cwiseProduct(result.raw_mass)).squaredNorm()
            * inverse_rows;
        result.mse_xmom =
            (operators.scale_momentum.cwiseProduct(
                result.raw_xmom)).squaredNorm() * inverse_rows;
        result.mse_ymom =
            (operators.scale_momentum.cwiseProduct(
                result.raw_ymom)).squaredNorm() * inverse_rows;
        result.weighted_loss =
            lambda_mass * result.mse_mass +
            lambda_xmom * result.mse_xmom +
            lambda_ymom * result.mse_ymom;

        if (!compute_adjoints)
            return result;

        result.coefficient_adjoints.resizeAndZero(coefficient_count);
        const Vector z_mass =
            (T(2) * lambda_mass * inverse_rows) *
            operators.scale_mass.cwiseProduct(
                operators.scale_mass.cwiseProduct(result.raw_mass));
        const Vector z_xmom =
            (T(2) * lambda_xmom * inverse_rows) *
            operators.scale_momentum.cwiseProduct(
                operators.scale_momentum.cwiseProduct(result.raw_xmom));
        const Vector z_ymom =
            (T(2) * lambda_ymom * inverse_rows) *
            operators.scale_momentum.cwiseProduct(
                operators.scale_momentum.cwiseProduct(result.raw_ymom));

        const Vector ex_mass = operators.Ex.transpose() * z_mass;
        const Vector ey_mass = operators.Ey.transpose() * z_mass;
        const Vector ex_x = operators.Ex.transpose() * z_xmom;
        const Vector ey_x = operators.Ey.transpose() * z_xmom;
        const Vector ex_y = operators.Ex.transpose() * z_ymom;
        const Vector ey_y = operators.Ey.transpose() * z_ymom;

        Vector adjoint_u = ex_mass;
        Vector adjoint_ux = Vector::Zero(sample_count);
        Vector adjoint_uy = Vector::Zero(sample_count);
        Vector adjoint_v = ey_mass;
        Vector adjoint_vx = Vector::Zero(sample_count);
        Vector adjoint_vy = Vector::Zero(sample_count);
        Vector adjoint_p = Vector::Zero(sample_count);

        adjoint_u += (T(2) * u).cwiseProduct(ex_x) + v.cwiseProduct(ey_x);
        adjoint_v += u.cwiseProduct(ey_x);
        adjoint_ux -= viscosity * ex_x;
        adjoint_uy -= viscosity * ey_x;
        adjoint_p += ex_x;

        adjoint_u += v.cwiseProduct(ex_y);
        adjoint_v +=
            u.cwiseProduct(ex_y) + (T(2) * v).cwiseProduct(ey_y);
        adjoint_vx -= viscosity * ex_y;
        adjoint_vy -= viscosity * ey_y;
        adjoint_p += ey_y;

        for (int sample = 0; sample < sample_count; ++sample)
        {
            scatterFaceAdjoint(
                operators, operators.face_samples[sample],
                adjoint_u(sample), adjoint_ux(sample), adjoint_uy(sample),
                result.coefficient_adjoints.u);
            scatterFaceAdjoint(
                operators, operators.face_samples[sample],
                adjoint_v(sample), adjoint_vx(sample), adjoint_vy(sample),
                result.coefficient_adjoints.v);
            scatterFaceAdjoint(
                operators, operators.face_samples[sample],
                adjoint_p(sample), T(0), T(0),
                result.coefficient_adjoints.p);
        }
        addStorageTranspose(
            operators, z_xmom, result.coefficient_adjoints.u);
        addStorageTranspose(
            operators, z_ymom, result.coefficient_adjoints.v);
        return result;
    }

private:
    struct GaussRule
    {
        std::vector<T> nodes;
        std::vector<T> weights;
    };

    struct FaceQuadrature
    {
        std::vector<int> samples;
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
                "Knot-span quadrature order must be 4 or 8");
        return rule;
    }

    static void validateKnotVector(
        const std::vector<T>& knots,
        int degree,
        int axis)
    {
        if (knots.size() < static_cast<std::size_t>(2 * degree + 2))
            throw std::invalid_argument(
                "Knot-span balance knot vector is too short");
        for (std::size_t index = 0; index < knots.size(); ++index)
        {
            if (!std::isfinite(knots[index]) ||
                knots[index] < T(0) || knots[index] > T(1) ||
                (index > 0 && knots[index] < knots[index - 1]))
                throw std::invalid_argument(
                    "Knot-span balance knot vector is invalid on axis " +
                    std::to_string(axis));
        }
        for (int index = 0; index <= degree; ++index)
            if (knots[index] != T(0) ||
                knots[knots.size() - 1 - index] != T(1))
                throw std::invalid_argument(
                    "Knot-span balance requires clamped [0,1] knots");
    }

    static void validateTimes(
        const std::vector<int>& indices,
        const std::vector<T>& time_axis,
        T physical_lower,
        T physical_upper,
        T minimum_time,
        const char* label)
    {
        if (indices.empty())
            throw std::invalid_argument(
                std::string("Knot-span ") + label +
                " time-index set is empty");
        const T tolerance =
            T(1.0e-6) * std::max<T>(T(1), std::abs(minimum_time));
        for (std::size_t position = 0; position < indices.size(); ++position)
        {
            const int index = indices[position];
            if (index < 0 || index >= static_cast<int>(time_axis.size()))
                throw std::invalid_argument(
                    std::string("Knot-span ") + label +
                    " time index is out of range");
            if (position > 0 && index <= indices[position - 1])
                throw std::invalid_argument(
                    std::string("Knot-span ") + label +
                    " time indices must be strictly increasing");
            if (time_axis[index] + tolerance < minimum_time ||
                time_axis[index] < physical_lower - tolerance ||
                time_axis[index] > physical_upper + tolerance)
                throw std::invalid_argument(
                    std::string("Knot-span ") + label +
                    " time index is outside the active physical domain");
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

    static NSKnotSpanGLBasis1D<T> basis(
        T parameter,
        const std::vector<T>& knots)
    {
        if (!std::isfinite(parameter) ||
            parameter < T(0) || parameter > T(1))
            throw std::out_of_range(
                "Knot-span basis parameter lies outside [0,1]");
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
                        "Knot-span basis encountered a zero denominator");
                const T temporary =
                    ndu[row][column - 1] / ndu[column][row];
                ndu[row][column] = saved + right[row + 1] * temporary;
                saved = left[column - row] * temporary;
            }
            ndu[column][column] = saved;
        }

        NSKnotSpanGLBasis1D<T> result;
        result.first_control = span - degree;
        result.parameter = parameter;
        for (int local = 0; local <= degree; ++local)
            result.value[local] = ndu[local][degree];

        T workspace[2][4] = {};
        for (int local = 0; local <= degree; ++local)
        {
            const int source = 0;
            const int destination = 1;
            workspace[source][0] = T(1);
            T value = T(0);
            const int reduced_local = local - 1;
            const int reduced_degree = degree - 1;
            if (local >= 1)
            {
                const T denominator =
                    ndu[reduced_degree + 1][reduced_local];
                if (denominator == T(0))
                    throw std::runtime_error(
                        "Knot-span derivative encountered a zero denominator");
                workspace[destination][0] =
                    workspace[source][0] / denominator;
                value = workspace[destination][0] *
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
                        "Knot-span derivative encountered a zero denominator");
                workspace[destination][offset] =
                    (workspace[source][offset] -
                     workspace[source][offset - 1]) / denominator;
                value += workspace[destination][offset] *
                    ndu[reduced_local + offset][reduced_degree];
            }
            if (local <= reduced_degree)
            {
                const T denominator =
                    ndu[reduced_degree + 1][local];
                if (denominator == T(0))
                    throw std::runtime_error(
                        "Knot-span derivative encountered a zero denominator");
                workspace[destination][1] =
                    -workspace[source][0] / denominator;
                value += workspace[destination][1] *
                    ndu[local][reduced_degree];
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
                    "Knot-span balance produced a nonpositive span");
            result.push_back(std::make_pair(
                boundaries[index], boundaries[index + 1]));
        }
        return result;
    }

    static int registerBasis(
        T parameter,
        const std::vector<T>& knots,
        std::map<T, int>& registry,
        std::vector<NSKnotSpanGLBasis1D<T> >& bases)
    {
        typename std::map<T, int>::const_iterator found =
            registry.find(parameter);
        if (found != registry.end())
            return found->second;
        const int index = static_cast<int>(bases.size());
        bases.push_back(basis(parameter, knots));
        registry.insert(std::make_pair(parameter, index));
        return index;
    }

    static NSKnotSpanGLIntegratedBasis1D<T> integratedBasis(
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
                const NSKnotSpanGLBasis1D<T> local = basis(parameter, knots);
                for (int offset = 0; offset < 4; ++offset)
                    dense[local.first_control + offset] +=
                        weight * local.value[offset];
            }
        }
        NSKnotSpanGLIntegratedBasis1D<T> result;
        const T tolerance =
            T(64) * std::numeric_limits<T>::epsilon() *
            std::max<T>(T(1), physical_extent);
        for (int control = 0; control < control_count; ++control)
            if (std::abs(dense[control]) > tolerance)
            {
                result.controls.push_back(control);
                result.weights.push_back(dense[control]);
            }
        return result;
    }

    static FaceQuadrature buildFace(
        int normal_axis,
        int fixed_partition_index,
        int tangential_partition_index,
        int t_basis,
        int parts_normal,
        int parts_tangential,
        const std::array<std::vector<T>, 3>& knots,
        const std::array<T, 3>& physical_extents,
        int quadrature_order,
        std::map<T, int>& x_registry,
        std::map<T, int>& y_registry,
        std::vector<NSKnotSpanGLBasis1D<T> >& x_bases,
        std::vector<NSKnotSpanGLBasis1D<T> >& y_bases,
        std::vector<NSKnotSpanGLFaceSample>& samples)
    {
        const int tangential_axis = normal_axis == 0 ? 1 : 0;
        const T fixed = static_cast<T>(fixed_partition_index) /
            static_cast<T>(parts_normal);
        const T lower = static_cast<T>(tangential_partition_index) /
            static_cast<T>(parts_tangential);
        const T upper = static_cast<T>(tangential_partition_index + 1) /
            static_cast<T>(parts_tangential);
        const std::vector<std::pair<T, T> > spans = clippedSpans(
            lower, upper, uniqueKnots(knots[tangential_axis]));
        const GaussRule rule = gaussRule(quadrature_order);
        FaceQuadrature result;
        for (std::size_t span_index = 0;
             span_index < spans.size(); ++span_index)
        {
            const T midpoint =
                T(0.5) * (spans[span_index].first + spans[span_index].second);
            const T half_width =
                T(0.5) * (spans[span_index].second - spans[span_index].first);
            for (std::size_t node = 0; node < rule.nodes.size(); ++node)
            {
                const T tangential =
                    midpoint + half_width * rule.nodes[node];
                const T x = normal_axis == 0 ? fixed : tangential;
                const T y = normal_axis == 0 ? tangential : fixed;
                NSKnotSpanGLFaceSample sample;
                sample.x_basis = registerBasis(
                    x, knots[0], x_registry, x_bases);
                sample.y_basis = registerBasis(
                    y, knots[1], y_registry, y_bases);
                sample.t_basis = t_basis;
                sample.normal_axis = normal_axis;
                sample.fixed_partition_index = fixed_partition_index;
                const int sample_index = static_cast<int>(samples.size());
                samples.push_back(sample);
                result.samples.push_back(sample_index);
                result.weights.push_back(
                    physical_extents[tangential_axis] *
                    half_width * rule.weights[node]);
            }
        }
        return result;
    }

    static NSKnotSpanGLBlockOperators<T> buildRows(
        const std::array<std::vector<T>, 3>& knots,
        const std::array<T, 3>& physical_lower,
        const std::array<T, 3>& physical_upper,
        const std::vector<T>& time_axis,
        const std::vector<int>& time_indices,
        int parts_x,
        int parts_y,
        int quadrature_order,
        bool check_closure,
        T closure_tolerance)
    {
        if (parts_x < 1 || parts_y < 1)
            throw std::invalid_argument(
                "Knot-span balance partition counts must be positive");
        NSKnotSpanGLBlockOperators<T> result;
        result.parts_x = parts_x;
        result.parts_y = parts_y;
        result.quadrature_order = quadrature_order;
        result.time_indices = time_indices;
        for (int axis = 0; axis < 3; ++axis)
        {
            result.control_counts[axis] =
                static_cast<int>(knots[axis].size()) - 4;
            result.physical_extents[axis] =
                physical_upper[axis] - physical_lower[axis];
        }

        for (int part = 0; part < parts_x; ++part)
            result.integrated_x.push_back(integratedBasis(
                static_cast<T>(part) / static_cast<T>(parts_x),
                static_cast<T>(part + 1) / static_cast<T>(parts_x),
                result.physical_extents[0], knots[0], quadrature_order));
        for (int part = 0; part < parts_y; ++part)
            result.integrated_y.push_back(integratedBasis(
                static_cast<T>(part) / static_cast<T>(parts_y),
                static_cast<T>(part + 1) / static_cast<T>(parts_y),
                result.physical_extents[1], knots[1], quadrature_order));

        std::map<T, int> x_registry;
        std::map<T, int> y_registry;
        std::map<T, int> t_registry;
        const int blocks = parts_x * parts_y;
        const int rows = blocks * static_cast<int>(time_indices.size());
        result.scale_mass.resize(rows);
        result.scale_momentum.resize(rows);
        std::vector<Triplet> ex_triplets;
        std::vector<Triplet> ey_triplets;

        for (std::size_t time_position = 0;
             time_position < time_indices.size(); ++time_position)
        {
            const T physical_time = time_axis[time_indices[time_position]];
            T parameter_time =
                (physical_time - physical_lower[2]) /
                result.physical_extents[2];
            if (parameter_time < T(0) &&
                parameter_time > -T(64) * std::numeric_limits<T>::epsilon())
                parameter_time = T(0);
            if (parameter_time > T(1) &&
                parameter_time < T(1) +
                    T(64) * std::numeric_limits<T>::epsilon())
                parameter_time = T(1);
            const int t_basis = registerBasis(
                parameter_time, knots[2], t_registry, result.t_bases);
            result.time_basis_indices.push_back(t_basis);

            std::vector<FaceQuadrature> vertical(
                static_cast<std::size_t>(parts_x + 1) * parts_y);
            std::vector<FaceQuadrature> horizontal(
                static_cast<std::size_t>(parts_y + 1) * parts_x);
            for (int fixed = 0; fixed <= parts_x; ++fixed)
                for (int segment = 0; segment < parts_y; ++segment)
                    vertical[static_cast<std::size_t>(fixed) * parts_y +
                             segment] = buildFace(
                        0, fixed, segment, t_basis,
                        parts_x, parts_y, knots, result.physical_extents,
                        quadrature_order, x_registry, y_registry,
                        result.x_bases, result.y_bases,
                        result.face_samples);
            for (int fixed = 0; fixed <= parts_y; ++fixed)
                for (int segment = 0; segment < parts_x; ++segment)
                    horizontal[static_cast<std::size_t>(fixed) * parts_x +
                               segment] = buildFace(
                        1, fixed, segment, t_basis,
                        parts_y, parts_x, knots, result.physical_extents,
                        quadrature_order, x_registry, y_registry,
                        result.x_bases, result.y_bases,
                        result.face_samples);

            for (int by = 0; by < parts_y; ++by)
                for (int bx = 0; bx < parts_x; ++bx)
                {
                    const int row =
                        static_cast<int>(time_position) * blocks +
                        by * parts_x + bx;
                    const T area =
                        result.physical_extents[0] /
                        static_cast<T>(parts_x) *
                        result.physical_extents[1] /
                        static_cast<T>(parts_y);
                    const T inverse_length = T(1) / std::sqrt(area);
                    result.scale_mass(row) = inverse_length;
                    result.scale_momentum(row) = inverse_length;

                    const FaceQuadrature& left =
                        vertical[static_cast<std::size_t>(bx) * parts_y + by];
                    const FaceQuadrature& right =
                        vertical[static_cast<std::size_t>(bx + 1) * parts_y + by];
                    for (std::size_t index = 0;
                         index < left.samples.size(); ++index)
                        ex_triplets.push_back(Triplet(
                            row, left.samples[index], -left.weights[index]));
                    for (std::size_t index = 0;
                         index < right.samples.size(); ++index)
                        ex_triplets.push_back(Triplet(
                            row, right.samples[index], right.weights[index]));

                    const FaceQuadrature& bottom =
                        horizontal[static_cast<std::size_t>(by) * parts_x + bx];
                    const FaceQuadrature& top =
                        horizontal[static_cast<std::size_t>(by + 1) * parts_x + bx];
                    for (std::size_t index = 0;
                         index < bottom.samples.size(); ++index)
                        ey_triplets.push_back(Triplet(
                            row, bottom.samples[index], -bottom.weights[index]));
                    for (std::size_t index = 0;
                         index < top.samples.size(); ++index)
                        ey_triplets.push_back(Triplet(
                            row, top.samples[index], top.weights[index]));
                }
        }

        const int sample_count =
            static_cast<int>(result.face_samples.size());
        result.Ex.resize(rows, sample_count);
        result.Ey.resize(rows, sample_count);
        result.Ex.setFromTriplets(ex_triplets.begin(), ex_triplets.end());
        result.Ey.setFromTriplets(ey_triplets.begin(), ey_triplets.end());
        result.Ex.makeCompressed();
        result.Ey.makeCompressed();

        const Vector ones = Vector::Ones(rows);
        const Vector sum_ex = result.Ex.transpose() * ones;
        const Vector sum_ey = result.Ey.transpose() * ones;
        T maximum_internal = T(0);
        for (int sample = 0; sample < sample_count; ++sample)
        {
            const NSKnotSpanGLFaceSample& face = result.face_samples[sample];
            const bool internal = face.normal_axis == 0
                ? (face.fixed_partition_index > 0 &&
                   face.fixed_partition_index < parts_x)
                : (face.fixed_partition_index > 0 &&
                   face.fixed_partition_index < parts_y);
            if (internal)
            {
                const T discrepancy = face.normal_axis == 0
                    ? std::abs(sum_ex(sample))
                    : std::abs(sum_ey(sample));
                maximum_internal = std::max(maximum_internal, discrepancy);
            }
        }
        result.maximum_internal_face_closure = maximum_internal;
        if (check_closure && maximum_internal > closure_tolerance)
            throw std::runtime_error(
                "Knot-span integral internal-face closure check failed");
        return result;
    }

    static int coefficientIndex(
        const NSKnotSpanGLBlockOperators<T>& operators,
        int x,
        int y,
        int t)
    {
        return x + operators.control_counts[0] *
            (y + operators.control_counts[1] * t);
    }

    static void evaluateFaceSample(
        const NSKnotSpanGLBlockOperators<T>& operators,
        const NSKnotSpanGLFaceSample& sample,
        const Vector& coefficients,
        T& value,
        T& derivative_x,
        T& derivative_y)
    {
        const NSKnotSpanGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const NSKnotSpanGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const NSKnotSpanGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        value = T(0);
        derivative_x = T(0);
        derivative_y = T(0);
        for (int local_t = 0; local_t < 4; ++local_t)
            for (int local_y = 0; local_y < 4; ++local_y)
                for (int local_x = 0; local_x < 4; ++local_x)
                {
                    const int index = coefficientIndex(
                        operators,
                        x.first_control + local_x,
                        y.first_control + local_y,
                        t.first_control + local_t);
                    const T coefficient = coefficients(index);
                    const T common = y.value[local_y] * t.value[local_t];
                    value += coefficient * x.value[local_x] * common;
                    derivative_x +=
                        coefficient * x.derivative[local_x] * common;
                    derivative_y += coefficient * x.value[local_x] *
                        y.derivative[local_y] * t.value[local_t];
                }
        derivative_x /= operators.physical_extents[0];
        derivative_y /= operators.physical_extents[1];
    }

    static Vector storageValues(
        const NSKnotSpanGLBlockOperators<T>& operators,
        const Vector& coefficients)
    {
        Vector result = Vector::Zero(operators.rows());
        const int blocks = operators.blocks();
        for (std::size_t time_position = 0;
             time_position < operators.time_indices.size(); ++time_position)
        {
            const NSKnotSpanGLBasis1D<T>& time =
                operators.t_bases[
                    operators.time_basis_indices[time_position]];
            for (int by = 0; by < operators.parts_y; ++by)
                for (int bx = 0; bx < operators.parts_x; ++bx)
                {
                    const int row =
                        static_cast<int>(time_position) * blocks +
                        by * operators.parts_x + bx;
                    const NSKnotSpanGLIntegratedBasis1D<T>& x =
                        operators.integrated_x[bx];
                    const NSKnotSpanGLIntegratedBasis1D<T>& y =
                        operators.integrated_y[by];
                    T value = T(0);
                    for (int local_t = 0; local_t < 4; ++local_t)
                    {
                        const T dt = time.derivative[local_t] /
                            operators.physical_extents[2];
                        for (std::size_t local_y = 0;
                             local_y < y.controls.size(); ++local_y)
                            for (std::size_t local_x = 0;
                                 local_x < x.controls.size(); ++local_x)
                            {
                                const int index = coefficientIndex(
                                    operators,
                                    x.controls[local_x],
                                    y.controls[local_y],
                                    time.first_control + local_t);
                                value += coefficients(index) * dt *
                                    x.weights[local_x] * y.weights[local_y];
                            }
                    }
                    result(row) = value;
                }
        }
        return result;
    }

    static void scatterFaceAdjoint(
        const NSKnotSpanGLBlockOperators<T>& operators,
        const NSKnotSpanGLFaceSample& sample,
        T adjoint_value,
        T adjoint_derivative_x,
        T adjoint_derivative_y,
        Vector& coefficient_adjoint)
    {
        const NSKnotSpanGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const NSKnotSpanGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const NSKnotSpanGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        for (int local_t = 0; local_t < 4; ++local_t)
            for (int local_y = 0; local_y < 4; ++local_y)
                for (int local_x = 0; local_x < 4; ++local_x)
                {
                    const int index = coefficientIndex(
                        operators,
                        x.first_control + local_x,
                        y.first_control + local_y,
                        t.first_control + local_t);
                    const T basis_value =
                        x.value[local_x] * y.value[local_y] *
                        t.value[local_t];
                    const T basis_dx =
                        x.derivative[local_x] * y.value[local_y] *
                        t.value[local_t] / operators.physical_extents[0];
                    const T basis_dy =
                        x.value[local_x] * y.derivative[local_y] *
                        t.value[local_t] / operators.physical_extents[1];
                    coefficient_adjoint(index) +=
                        adjoint_value * basis_value +
                        adjoint_derivative_x * basis_dx +
                        adjoint_derivative_y * basis_dy;
                }
    }

    static void addStorageTranspose(
        const NSKnotSpanGLBlockOperators<T>& operators,
        const Vector& row_adjoint,
        Vector& coefficient_adjoint)
    {
        const int blocks = operators.blocks();
        for (std::size_t time_position = 0;
             time_position < operators.time_indices.size(); ++time_position)
        {
            const NSKnotSpanGLBasis1D<T>& time =
                operators.t_bases[
                    operators.time_basis_indices[time_position]];
            for (int by = 0; by < operators.parts_y; ++by)
                for (int bx = 0; bx < operators.parts_x; ++bx)
                {
                    const int row =
                        static_cast<int>(time_position) * blocks +
                        by * operators.parts_x + bx;
                    const T multiplier = row_adjoint(row);
                    const NSKnotSpanGLIntegratedBasis1D<T>& x =
                        operators.integrated_x[bx];
                    const NSKnotSpanGLIntegratedBasis1D<T>& y =
                        operators.integrated_y[by];
                    for (int local_t = 0; local_t < 4; ++local_t)
                    {
                        const T dt = time.derivative[local_t] /
                            operators.physical_extents[2];
                        for (std::size_t local_y = 0;
                             local_y < y.controls.size(); ++local_y)
                            for (std::size_t local_x = 0;
                                 local_x < x.controls.size(); ++local_x)
                            {
                                const int index = coefficientIndex(
                                    operators,
                                    x.controls[local_x],
                                    y.controls[local_y],
                                    time.first_control + local_t);
                                coefficient_adjoint(index) += multiplier * dt *
                                    x.weights[local_x] * y.weights[local_y];
                            }
                    }
                }
        }
    }

    static void requireCoefficients(
        const Vector& coefficients,
        int expected,
        const char* label)
    {
        if (coefficients.size() != expected || !coefficients.allFinite())
            throw std::invalid_argument(
                std::string("Invalid knot-span coefficients for ") + label);
    }
};

} // namespace mfa

#endif // MFA_NS_KNOT_SPAN_INTEGRAL_BALANCE_HPP
