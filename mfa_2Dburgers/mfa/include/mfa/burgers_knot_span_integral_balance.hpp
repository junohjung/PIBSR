//--------------------------------------------------------------
// Representation-consistent finite-region balances for the 2-D
// advective Burgers equations represented by cubic, nonrational,
// tensor-product B-splines.
//
// For each region K, the operator evaluates
//
//   b_u = int_K u_t
//       + int_dK [(u^2/2, uv) - nu grad(u)] . n
//       - int_K u v_y,
//
//   b_v = int_K v_t
//       + int_dK [(uv, v^2/2) - nu grad(v)] . n
//       - int_K v u_x.
//
// The volume corrections are essential: without them, the operator
// would enforce conservative Burgers rather than the advective form
// used by the manuscript.  Volume and face integrals are split at all
// knot spans and evaluated with Gauss--Legendre quadrature.  Every
// internal face is assembled once and referenced with opposite signs
// by its neighbors, so shared-face cancellation is algebraic.
//--------------------------------------------------------------

#ifndef MFA_BURGERS_KNOT_SPAN_INTEGRAL_BALANCE_HPP
#define MFA_BURGERS_KNOT_SPAN_INTEGRAL_BALANCE_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfa
{

template <typename T>
inline T burgersIntegralStrictScalar(
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

inline int burgersIntegralStrictInteger(
    const char* name,
    int default_value,
    int minimum)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        return default_value;
    std::istringstream stream(text);
    int value = 0;
    char trailing = '\0';
    if (!(stream >> value) || (stream >> trailing) || value < minimum)
        throw std::invalid_argument(
            std::string("Malformed integer in ") + name + ": " + text);
    return value;
}

inline bool burgersIntegralStrictBoolean(
    const char* name,
    bool default_value)
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

inline std::string burgersIntegralRequiredPath(const char* name)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        throw std::invalid_argument(
            std::string(name) +
            " is required when a Burgers integral weight is positive or "
            "diagnostics are enabled");
    return std::string(text);
}

inline std::vector<int> burgersIntegralReadIndices(
    const std::string& path,
    const char* label)
{
    std::ifstream stream(path.c_str());
    if (!stream)
        throw std::runtime_error(std::string("Cannot open ") + label + ": " + path);
    std::vector<int> indices;
    std::set<int> seen;
    long long value = -1;
    while (stream >> value)
    {
        if (value < 0 || value > std::numeric_limits<int>::max())
            throw std::runtime_error(std::string("Out-of-range integer in ") + label);
        const int index = static_cast<int>(value);
        if (!seen.insert(index).second)
            throw std::runtime_error(std::string("Duplicate time index in ") + label);
        indices.push_back(index);
    }
    if (!stream.eof())
        throw std::runtime_error(std::string("Malformed time index in ") + label);
    if (indices.empty())
        throw std::runtime_error(std::string("No time indices in ") + label);
    if (!std::is_sorted(indices.begin(), indices.end()))
        throw std::runtime_error(
            std::string(label) + " must contain sorted canonical indices");
    return indices;
}

template <typename T>
struct BurgersIntegralBalanceConfig
{
    T lambda_u;
    T lambda_v;
    int parts_x;
    int parts_y;
    bool diagnostics;
    bool check_closure;
    T closure_tolerance;
    std::vector<int> fit_time_indices;
    std::vector<int> validation_time_indices;

    BurgersIntegralBalanceConfig()
        : lambda_u(T(0)),
          lambda_v(T(0)),
          parts_x(4),
          parts_y(4),
          diagnostics(false),
          check_closure(true),
          closure_tolerance(T(1.0e-12))
    {
    }

    bool enabled() const
    {
        return lambda_u > T(0) || lambda_v > T(0);
    }

    bool operatorsRequired() const
    {
        return enabled() || diagnostics;
    }

    static BurgersIntegralBalanceConfig fromEnvironment()
    {
        BurgersIntegralBalanceConfig config;
        config.lambda_u = burgersIntegralStrictScalar<T>(
            "MFA_BURGERS_LAMBDA_INT_U", T(0));
        config.lambda_v = burgersIntegralStrictScalar<T>(
            "MFA_BURGERS_LAMBDA_INT_V", T(0));
        config.diagnostics = burgersIntegralStrictBoolean(
            "MFA_BURGERS_INT_DIAGNOSTICS", false);

        // This early return is the strict zero-weight fast path.  In the
        // default configuration no index file is read and no quadrature
        // operator is constructed or evaluated.
        if (!config.operatorsRequired())
            return config;

        config.parts_x = burgersIntegralStrictInteger(
            "MFA_BURGERS_INT_PARTS_X", 4, 1);
        config.parts_y = burgersIntegralStrictInteger(
            "MFA_BURGERS_INT_PARTS_Y", 4, 1);
        config.check_closure = burgersIntegralStrictBoolean(
            "MFA_BURGERS_INT_CHECK_CLOSURE", true);
        config.closure_tolerance = burgersIntegralStrictScalar<T>(
            "MFA_BURGERS_INT_CLOSURE_TOL", T(1.0e-12));
        if (!(config.closure_tolerance > T(0)))
            throw std::invalid_argument(
                "MFA_BURGERS_INT_CLOSURE_TOL must be positive");

        const char* quadrature = std::getenv("MFA_BURGERS_INT_QUADRATURE");
        if (quadrature && *quadrature && std::string(quadrature) != "gl4")
            throw std::invalid_argument(
                "MFA_BURGERS_INT_QUADRATURE must be exactly gl4; GL8 is "
                "reserved for independent post-fit evaluation");

        config.fit_time_indices = burgersIntegralReadIndices(
            burgersIntegralRequiredPath(
                "MFA_BURGERS_INT_FIT_TIME_INDEX_FILE"),
            "MFA_BURGERS_INT_FIT_TIME_INDEX_FILE");
        config.validation_time_indices = burgersIntegralReadIndices(
            burgersIntegralRequiredPath(
                "MFA_BURGERS_INT_VALID_TIME_INDEX_FILE"),
            "MFA_BURGERS_INT_VALID_TIME_INDEX_FILE");

        const std::set<int> fit(
            config.fit_time_indices.begin(), config.fit_time_indices.end());
        for (std::size_t i = 0; i < config.validation_time_indices.size(); ++i)
            if (fit.count(config.validation_time_indices[i]))
                throw std::invalid_argument(
                    "Burgers integral fit and validation time indices overlap");
        return config;
    }
};

template <typename T>
struct BurgersGLBasis1D
{
    int first_control;
    T parameter;
    std::array<T, 4> value;
    std::array<T, 4> derivative;
    std::array<T, 4> second_derivative;

    BurgersGLBasis1D() : first_control(0), parameter(T(0))
    {
        value.fill(T(0));
        derivative.fill(T(0));
        second_derivative.fill(T(0));
    }
};

struct BurgersGLSample
{
    int x_basis;
    int y_basis;
    int t_basis;
    int normal_axis;
    int fixed_partition_index;

    BurgersGLSample()
        : x_basis(0), y_basis(0), t_basis(0),
          normal_axis(-1), fixed_partition_index(-1)
    {
    }
};

template <typename T>
struct BurgersGLBlockOperators
{
    typedef Eigen::SparseMatrix<T> SparseMatrix;
    SparseMatrix V;
    SparseMatrix Ex;
    SparseMatrix Ey;
    Eigen::Matrix<T, Eigen::Dynamic, 1> scale;
    std::vector<BurgersGLSample> face_samples;
    std::vector<BurgersGLSample> volume_samples;
    std::vector<BurgersGLBasis1D<T> > x_bases;
    std::vector<BurgersGLBasis1D<T> > y_bases;
    std::vector<BurgersGLBasis1D<T> > t_bases;
    std::vector<int> time_indices;
    std::array<int, 3> control_counts;
    std::array<T, 3> physical_extents;
    int parts_x;
    int parts_y;
    int quadrature_order;
    T maximum_internal_face_closure;
    T maximum_volume_measure_error;

    BurgersGLBlockOperators()
        : parts_x(0), parts_y(0), quadrature_order(0),
          maximum_internal_face_closure(T(0)),
          maximum_volume_measure_error(T(0))
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
    std::size_t volumeSampleCount() const { return volume_samples.size(); }
};

template <typename T>
struct BurgersGLBalanceOperators
{
    BurgersGLBlockOperators<T> fit;
    BurgersGLBlockOperators<T> validation;
    std::array<T, 3> physical_lower;
    std::array<T, 3> physical_upper;
    std::vector<T> x_boundaries;
    std::vector<T> y_boundaries;
};

template <typename T>
struct BurgersGLCoefficientAdjoints
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector u;
    Vector v;

    void resizeAndZero(int count)
    {
        u = Vector::Zero(count);
        v = Vector::Zero(count);
    }
};

template <typename T>
struct BurgersGLBalanceEvaluation
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector raw_u;
    Vector raw_v;
    Vector direct_strong_u;
    Vector direct_strong_v;
    Vector storage_u;
    Vector storage_v;
    Vector advective_face_u;
    Vector advective_face_v;
    Vector volume_correction_u;
    Vector volume_correction_v;
    Vector diffusion_face_u;
    Vector diffusion_face_v;
    T mse_u;
    T mse_v;
    T weighted_loss;
    T maximum_flux_strong_mismatch;
    BurgersGLCoefficientAdjoints<T> coefficient_adjoints;

    BurgersGLBalanceEvaluation()
        : mse_u(T(0)), mse_v(T(0)), weighted_loss(T(0)),
          maximum_flux_strong_mismatch(T(0))
    {
    }
};

template <typename T>
class BurgersKnotSpanGLBalance
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Triplet<T> Triplet;

    static BurgersGLBalanceOperators<T> build(
        const std::array<std::vector<T>, 3>& knots,
        const std::array<int, 3>& degrees,
        const std::array<T, 3>& physical_lower,
        const std::array<T, 3>& physical_upper,
        const std::vector<T>& time_axis,
        const BurgersIntegralBalanceConfig<T>& config,
        int quadrature_order)
    {
        if (!config.operatorsRequired())
            throw std::logic_error(
                "BurgersKnotSpanGLBalance::build called while disabled");
        if (quadrature_order != 4 && quadrature_order != 8)
            throw std::invalid_argument(
                "Burgers knot-span quadrature order must be 4 or 8");
        for (int axis = 0; axis < 3; ++axis)
        {
            if (degrees[axis] != 3)
                throw std::invalid_argument(
                    "Burgers knot-span balance requires cubic splines");
            validateKnotVector(knots[axis], degrees[axis], axis);
            if (!std::isfinite(physical_lower[axis]) ||
                !std::isfinite(physical_upper[axis]) ||
                !(physical_upper[axis] > physical_lower[axis]))
                throw std::invalid_argument(
                    "Burgers knot-span physical bounds are invalid");
        }
        if (time_axis.empty())
            throw std::invalid_argument(
                "Burgers knot-span balance requires a nonempty time axis");
        for (std::size_t index = 0; index < time_axis.size(); ++index)
            if (!std::isfinite(time_axis[index]) ||
                (index > 0 && !(time_axis[index] > time_axis[index - 1])))
                throw std::invalid_argument(
                    "Burgers time axis must be finite and strictly increasing");
        validateTimes(config.fit_time_indices, time_axis, "fit");
        validateTimes(config.validation_time_indices, time_axis, "validation");

        BurgersGLBalanceOperators<T> result;
        result.physical_lower = physical_lower;
        result.physical_upper = physical_upper;
        result.fit = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.fit_time_indices, config.parts_x, config.parts_y,
            quadrature_order, config.check_closure,
            config.closure_tolerance);
        result.validation = buildRows(
            knots, physical_lower, physical_upper, time_axis,
            config.validation_time_indices, config.parts_x, config.parts_y,
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

    static BurgersGLBalanceEvaluation<T> evaluate(
        const BurgersGLBlockOperators<T>& operators,
        const Vector& coefficients_u,
        const Vector& coefficients_v,
        T viscosity,
        T lambda_u,
        T lambda_v,
        bool compute_adjoints,
        bool compute_form_diagnostic = true)
    {
        const int coefficient_count = operators.coefficientCount();
        requireCoefficients(coefficients_u, coefficient_count, "u");
        requireCoefficients(coefficients_v, coefficient_count, "v");
        if (operators.rows() <= 0)
            throw std::invalid_argument(
                "Burgers knot-span balance requires nonempty rows");
        if (!(viscosity > T(0)) || !std::isfinite(viscosity))
            throw std::invalid_argument(
                "Burgers knot-span viscosity must be positive and finite");
        if (lambda_u < T(0) || lambda_v < T(0) ||
            !std::isfinite(lambda_u) || !std::isfinite(lambda_v))
            throw std::invalid_argument(
                "Burgers integral weights must be finite and nonnegative");

        const int face_count = static_cast<int>(operators.face_samples.size());
        Vector uf(face_count), ufx(face_count), ufy(face_count);
        Vector vf(face_count), vfx(face_count), vfy(face_count);
        for (int sample = 0; sample < face_count; ++sample)
        {
            evaluateFaceSample(
                operators, operators.face_samples[sample], coefficients_u,
                uf(sample), ufx(sample), ufy(sample));
            evaluateFaceSample(
                operators, operators.face_samples[sample], coefficients_v,
                vf(sample), vfx(sample), vfy(sample));
        }

        const int volume_count =
            static_cast<int>(operators.volume_samples.size());
        Vector u(volume_count), ux(volume_count), ut(volume_count);
        Vector v(volume_count), vy(volume_count), vt(volume_count);
        Vector uy, uxx, uyy, vx, vxx, vyy;
        if (compute_form_diagnostic)
        {
            uy.resize(volume_count);
            uxx.resize(volume_count);
            uyy.resize(volume_count);
            vx.resize(volume_count);
            vxx.resize(volume_count);
            vyy.resize(volume_count);
            for (int sample = 0; sample < volume_count; ++sample)
            {
                evaluateVolumeSample(
                    operators, operators.volume_samples[sample], coefficients_u,
                    u(sample), ux(sample), uy(sample), ut(sample),
                    uxx(sample), uyy(sample));
                evaluateVolumeSample(
                    operators, operators.volume_samples[sample], coefficients_v,
                    v(sample), vx(sample), vy(sample), vt(sample),
                    vxx(sample), vyy(sample));
            }
        }
        else
        {
            for (int sample = 0; sample < volume_count; ++sample)
            {
                T unused_y = T(0);
                T unused_xx = T(0);
                T unused_yy = T(0);
                evaluateVolumeSample(
                    operators, operators.volume_samples[sample], coefficients_u,
                    u(sample), ux(sample), unused_y, ut(sample),
                    unused_xx, unused_yy);
                T unused_x = T(0);
                evaluateVolumeSample(
                    operators, operators.volume_samples[sample], coefficients_v,
                    v(sample), unused_x, vy(sample), vt(sample),
                    unused_xx, unused_yy);
            }
        }

        BurgersGLBalanceEvaluation<T> result;
        result.storage_u = operators.V * ut;
        result.storage_v = operators.V * vt;
        result.advective_face_u =
            operators.Ex * (T(0.5) * uf.cwiseProduct(uf)) +
            operators.Ey * (uf.cwiseProduct(vf));
        result.advective_face_v =
            operators.Ex * (uf.cwiseProduct(vf)) +
            operators.Ey * (T(0.5) * vf.cwiseProduct(vf));
        result.volume_correction_u =
            -operators.V * u.cwiseProduct(vy);
        result.volume_correction_v =
            -operators.V * v.cwiseProduct(ux);
        result.diffusion_face_u = -viscosity *
            (operators.Ex * ufx + operators.Ey * ufy);
        result.diffusion_face_v = -viscosity *
            (operators.Ex * vfx + operators.Ey * vfy);
        result.raw_u = result.storage_u + result.advective_face_u +
            result.volume_correction_u + result.diffusion_face_u;
        result.raw_v = result.storage_v + result.advective_face_v +
            result.volume_correction_v + result.diffusion_face_v;

        if (compute_form_diagnostic)
        {
            result.direct_strong_u = operators.V *
                (ut + u.cwiseProduct(ux) + v.cwiseProduct(uy) -
                 viscosity * (uxx + uyy));
            result.direct_strong_v = operators.V *
                (vt + u.cwiseProduct(vx) + v.cwiseProduct(vy) -
                 viscosity * (vxx + vyy));
            result.maximum_flux_strong_mismatch = std::max(
                (result.raw_u - result.direct_strong_u)
                    .template lpNorm<Eigen::Infinity>(),
                (result.raw_v - result.direct_strong_v)
                    .template lpNorm<Eigen::Infinity>());
        }
        else
            result.maximum_flux_strong_mismatch =
                std::numeric_limits<T>::quiet_NaN();

        const T inverse_rows = T(1) / static_cast<T>(operators.rows());
        result.mse_u =
            (operators.scale.cwiseProduct(result.raw_u)).squaredNorm() *
            inverse_rows;
        result.mse_v =
            (operators.scale.cwiseProduct(result.raw_v)).squaredNorm() *
            inverse_rows;
        result.weighted_loss =
            lambda_u * result.mse_u + lambda_v * result.mse_v;

        if (!compute_adjoints)
            return result;

        result.coefficient_adjoints.resizeAndZero(coefficient_count);
        const Vector z_u = (T(2) * lambda_u * inverse_rows) *
            operators.scale.cwiseProduct(
                operators.scale.cwiseProduct(result.raw_u));
        const Vector z_v = (T(2) * lambda_v * inverse_rows) *
            operators.scale.cwiseProduct(
                operators.scale.cwiseProduct(result.raw_v));

        const Vector ex_u = operators.Ex.transpose() * z_u;
        const Vector ey_u = operators.Ey.transpose() * z_u;
        const Vector ex_v = operators.Ex.transpose() * z_v;
        const Vector ey_v = operators.Ey.transpose() * z_v;
        Vector adjoint_uf =
            uf.cwiseProduct(ex_u) +
            vf.cwiseProduct(ey_u + ex_v);
        Vector adjoint_ufx = -viscosity * ex_u;
        Vector adjoint_ufy = -viscosity * ey_u;
        Vector adjoint_vf =
            uf.cwiseProduct(ey_u + ex_v) +
            vf.cwiseProduct(ey_v);
        Vector adjoint_vfx = -viscosity * ex_v;
        Vector adjoint_vfy = -viscosity * ey_v;
        for (int sample = 0; sample < face_count; ++sample)
        {
            scatterFaceAdjoint(
                operators, operators.face_samples[sample],
                adjoint_uf(sample), adjoint_ufx(sample), adjoint_ufy(sample),
                result.coefficient_adjoints.u);
            scatterFaceAdjoint(
                operators, operators.face_samples[sample],
                adjoint_vf(sample), adjoint_vfx(sample), adjoint_vfy(sample),
                result.coefficient_adjoints.v);
        }

        const Vector volume_u = operators.V.transpose() * z_u;
        const Vector volume_v = operators.V.transpose() * z_v;
        for (int sample = 0; sample < volume_count; ++sample)
        {
            scatterVolumeAdjoint(
                operators, operators.volume_samples[sample],
                -vy(sample) * volume_u(sample),
                -v(sample) * volume_v(sample), T(0), volume_u(sample),
                result.coefficient_adjoints.u);
            scatterVolumeAdjoint(
                operators, operators.volume_samples[sample],
                -ux(sample) * volume_v(sample),
                T(0), -u(sample) * volume_u(sample), volume_v(sample),
                result.coefficient_adjoints.v);
        }
        return result;
    }

    // Aggregate a q_x by q_y raw balance to a coarser partition.  This is
    // used for q=1/q=2 diagnostics without constructing another quadrature
    // lattice.  The returned values are unscaled physical balances.
    static Vector aggregateRaw(
        const BurgersGLBlockOperators<T>& operators,
        const Vector& raw,
        int coarse_parts_x,
        int coarse_parts_y)
    {
        if (raw.size() != operators.rows() || coarse_parts_x < 1 ||
            coarse_parts_y < 1 ||
            operators.parts_x % coarse_parts_x != 0 ||
            operators.parts_y % coarse_parts_y != 0)
            throw std::invalid_argument(
                "Invalid Burgers balance aggregation request");
        const int fine_per_x = operators.parts_x / coarse_parts_x;
        const int fine_per_y = operators.parts_y / coarse_parts_y;
        const int coarse_blocks = coarse_parts_x * coarse_parts_y;
        Vector result = Vector::Zero(
            static_cast<int>(operators.time_indices.size()) * coarse_blocks);
        for (std::size_t time = 0; time < operators.time_indices.size(); ++time)
            for (int fy = 0; fy < operators.parts_y; ++fy)
                for (int fx = 0; fx < operators.parts_x; ++fx)
                {
                    const int fine_row = static_cast<int>(time) *
                        operators.blocks() + fy * operators.parts_x + fx;
                    const int cx = fx / fine_per_x;
                    const int cy = fy / fine_per_y;
                    const int coarse_row = static_cast<int>(time) *
                        coarse_blocks + cy * coarse_parts_x + cx;
                    result(coarse_row) += raw(fine_row);
                }
        return result;
    }

    static T scaledMSE(
        const Vector& raw,
        int time_count,
        int parts_x,
        int parts_y,
        T domain_area)
    {
        const int rows = time_count * parts_x * parts_y;
        if (raw.size() != rows || rows <= 0 || !(domain_area > T(0)))
            throw std::invalid_argument(
                "Invalid Burgers balance MSE arguments");
        const T block_area = domain_area /
            static_cast<T>(parts_x * parts_y);
        return raw.squaredNorm() /
            (block_area * static_cast<T>(rows));
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
                T(-0.86113631159405257522), T(-0.33998104358485626480),
                T( 0.33998104358485626480), T( 0.86113631159405257522)};
            const T weights[] = {
                T(0.34785484513745385737), T(0.65214515486254614263),
                T(0.65214515486254614263), T(0.34785484513745385737)};
            rule.nodes.assign(nodes, nodes + 4);
            rule.weights.assign(weights, weights + 4);
        }
        else if (order == 8)
        {
            const T nodes[] = {
                T(-0.96028985649753623168), T(-0.79666647741362673959),
                T(-0.52553240991632898582), T(-0.18343464249564980494),
                T( 0.18343464249564980494), T( 0.52553240991632898582),
                T( 0.79666647741362673959), T( 0.96028985649753623168)};
            const T weights[] = {
                T(0.10122853629037625915), T(0.22238103445337447054),
                T(0.31370664587788728734), T(0.36268378337836198297),
                T(0.36268378337836198297), T(0.31370664587788728734),
                T(0.22238103445337447054), T(0.10122853629037625915)};
            rule.nodes.assign(nodes, nodes + 8);
            rule.weights.assign(weights, weights + 8);
        }
        else
            throw std::invalid_argument(
                "Burgers knot-span quadrature order must be 4 or 8");
        return rule;
    }

    static void validateKnotVector(
        const std::vector<T>& knots,
        int degree,
        int axis)
    {
        if (knots.size() < static_cast<std::size_t>(2 * degree + 2))
            throw std::invalid_argument(
                "Burgers knot-span knot vector is too short");
        for (std::size_t index = 0; index < knots.size(); ++index)
            if (!std::isfinite(knots[index]) || knots[index] < T(0) ||
                knots[index] > T(1) ||
                (index > 0 && knots[index] < knots[index - 1]))
                throw std::invalid_argument(
                    "Invalid Burgers knot vector on axis " +
                    std::to_string(axis));
        for (int index = 0; index <= degree; ++index)
            if (knots[index] != T(0) ||
                knots[knots.size() - 1 - index] != T(1))
                throw std::invalid_argument(
                    "Burgers knot-span balance requires clamped [0,1] knots");
    }

    static void validateTimes(
        const std::vector<int>& indices,
        const std::vector<T>& time_axis,
        const char* label)
    {
        if (indices.empty())
            throw std::invalid_argument(
                std::string("Burgers ") + label + " time-index set is empty");
        for (std::size_t position = 0; position < indices.size(); ++position)
        {
            const int index = indices[position];
            if (index < 0 || index >= static_cast<int>(time_axis.size()))
                throw std::invalid_argument(
                    std::string("Burgers ") + label +
                    " time index is out of range");
            if (position > 0 && index <= indices[position - 1])
                throw std::invalid_argument(
                    std::string("Burgers ") + label +
                    " time indices must be strictly increasing");
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
        while (parameter < knots[middle] || parameter >= knots[middle + 1])
        {
            if (parameter < knots[middle])
                high = middle;
            else
                low = middle;
            middle = (low + high) / 2;
        }
        return middle;
    }

    static BurgersGLBasis1D<T> basis(
        T parameter,
        const std::vector<T>& knots)
    {
        if (!std::isfinite(parameter) || parameter < T(0) || parameter > T(1))
            throw std::out_of_range(
                "Burgers knot-span basis parameter lies outside [0,1]");
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
                        "Burgers knot-span basis encountered a zero denominator");
                const T temporary = ndu[row][column - 1] / ndu[column][row];
                ndu[row][column] = saved + right[row + 1] * temporary;
                saved = left[column - row] * temporary;
            }
            ndu[column][column] = saved;
        }

        T derivatives[3][4] = {};
        for (int local = 0; local <= degree; ++local)
            derivatives[0][local] = ndu[local][degree];
        T workspace[2][4] = {};
        for (int local = 0; local <= degree; ++local)
        {
            int source = 0;
            int destination = 1;
            workspace[0][0] = T(1);
            for (int order = 1; order <= 2; ++order)
            {
                T value = T(0);
                const int reduced_local = local - order;
                const int reduced_degree = degree - order;
                if (local >= order)
                {
                    const T denominator =
                        ndu[reduced_degree + 1][reduced_local];
                    if (denominator == T(0))
                        throw std::runtime_error(
                            "Burgers basis derivative has zero denominator");
                    workspace[destination][0] =
                        workspace[source][0] / denominator;
                    value = workspace[destination][0] *
                        ndu[reduced_local][reduced_degree];
                }
                const int first = reduced_local >= -1 ? 1 : -reduced_local;
                const int last = local - 1 <= reduced_degree
                    ? order - 1 : degree - local;
                for (int offset = first; offset <= last; ++offset)
                {
                    const T denominator =
                        ndu[reduced_degree + 1][reduced_local + offset];
                    if (denominator == T(0))
                        throw std::runtime_error(
                            "Burgers basis derivative has zero denominator");
                    workspace[destination][offset] =
                        (workspace[source][offset] -
                         workspace[source][offset - 1]) / denominator;
                    value += workspace[destination][offset] *
                        ndu[reduced_local + offset][reduced_degree];
                }
                if (local <= reduced_degree)
                {
                    const T denominator = ndu[reduced_degree + 1][local];
                    if (denominator == T(0))
                        throw std::runtime_error(
                            "Burgers basis derivative has zero denominator");
                    workspace[destination][order] =
                        -workspace[source][order - 1] / denominator;
                    value += workspace[destination][order] *
                        ndu[local][reduced_degree];
                }
                derivatives[order][local] = value;
                std::swap(source, destination);
            }
        }
        for (int local = 0; local <= degree; ++local)
        {
            derivatives[1][local] *= T(degree);
            derivatives[2][local] *= T(degree * (degree - 1));
        }

        BurgersGLBasis1D<T> result;
        result.first_control = span - degree;
        result.parameter = parameter;
        for (int local = 0; local <= degree; ++local)
        {
            result.value[local] = derivatives[0][local];
            result.derivative[local] = derivatives[1][local];
            result.second_derivative[local] = derivatives[2][local];
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
                    "Burgers balance produced a nonpositive span");
            result.push_back(std::make_pair(
                boundaries[index], boundaries[index + 1]));
        }
        return result;
    }

    static int registerBasis(
        T parameter,
        const std::vector<T>& knots,
        std::map<T, int>& registry,
        std::vector<BurgersGLBasis1D<T> >& bases)
    {
        typename std::map<T, int>::const_iterator found = registry.find(parameter);
        if (found != registry.end())
            return found->second;
        const int index = static_cast<int>(bases.size());
        bases.push_back(basis(parameter, knots));
        registry.insert(std::make_pair(parameter, index));
        return index;
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
        std::vector<BurgersGLBasis1D<T> >& x_bases,
        std::vector<BurgersGLBasis1D<T> >& y_bases,
        std::vector<BurgersGLSample>& samples)
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
        for (std::size_t span_index = 0; span_index < spans.size(); ++span_index)
        {
            const T midpoint = T(0.5) *
                (spans[span_index].first + spans[span_index].second);
            const T half_width = T(0.5) *
                (spans[span_index].second - spans[span_index].first);
            for (std::size_t node = 0; node < rule.nodes.size(); ++node)
            {
                const T tangential = midpoint + half_width * rule.nodes[node];
                const T x = normal_axis == 0 ? fixed : tangential;
                const T y = normal_axis == 0 ? tangential : fixed;
                BurgersGLSample sample;
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
                    physical_extents[tangential_axis] * half_width *
                    rule.weights[node]);
            }
        }
        return result;
    }

    static void buildVolume(
        int part_x,
        int part_y,
        int t_basis,
        int row,
        int parts_x,
        int parts_y,
        const std::array<std::vector<T>, 3>& knots,
        const std::array<T, 3>& physical_extents,
        int quadrature_order,
        std::map<T, int>& x_registry,
        std::map<T, int>& y_registry,
        std::vector<BurgersGLBasis1D<T> >& x_bases,
        std::vector<BurgersGLBasis1D<T> >& y_bases,
        std::vector<BurgersGLSample>& samples,
        std::vector<Triplet>& triplets)
    {
        const T x_lower = static_cast<T>(part_x) / static_cast<T>(parts_x);
        const T x_upper = static_cast<T>(part_x + 1) / static_cast<T>(parts_x);
        const T y_lower = static_cast<T>(part_y) / static_cast<T>(parts_y);
        const T y_upper = static_cast<T>(part_y + 1) / static_cast<T>(parts_y);
        const std::vector<std::pair<T, T> > x_spans = clippedSpans(
            x_lower, x_upper, uniqueKnots(knots[0]));
        const std::vector<std::pair<T, T> > y_spans = clippedSpans(
            y_lower, y_upper, uniqueKnots(knots[1]));
        const GaussRule rule = gaussRule(quadrature_order);
        for (std::size_t sy = 0; sy < y_spans.size(); ++sy)
        {
            const T y_mid = T(0.5) *
                (y_spans[sy].first + y_spans[sy].second);
            const T y_half = T(0.5) *
                (y_spans[sy].second - y_spans[sy].first);
            for (std::size_t sx = 0; sx < x_spans.size(); ++sx)
            {
                const T x_mid = T(0.5) *
                    (x_spans[sx].first + x_spans[sx].second);
                const T x_half = T(0.5) *
                    (x_spans[sx].second - x_spans[sx].first);
                for (std::size_t jy = 0; jy < rule.nodes.size(); ++jy)
                    for (std::size_t ix = 0; ix < rule.nodes.size(); ++ix)
                    {
                        BurgersGLSample sample;
                        sample.x_basis = registerBasis(
                            x_mid + x_half * rule.nodes[ix],
                            knots[0], x_registry, x_bases);
                        sample.y_basis = registerBasis(
                            y_mid + y_half * rule.nodes[jy],
                            knots[1], y_registry, y_bases);
                        sample.t_basis = t_basis;
                        const int sample_index = static_cast<int>(samples.size());
                        samples.push_back(sample);
                        const T weight = physical_extents[0] *
                            physical_extents[1] * x_half * y_half *
                            rule.weights[ix] * rule.weights[jy];
                        triplets.push_back(Triplet(row, sample_index, weight));
                    }
            }
        }
    }

    static BurgersGLBlockOperators<T> buildRows(
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
                "Burgers partition counts must be positive");
        BurgersGLBlockOperators<T> result;
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

        std::map<T, int> x_registry;
        std::map<T, int> y_registry;
        std::map<T, int> t_registry;
        const int blocks = parts_x * parts_y;
        const int rows = blocks * static_cast<int>(time_indices.size());
        result.scale.resize(rows);
        std::vector<Triplet> ex_triplets;
        std::vector<Triplet> ey_triplets;
        std::vector<Triplet> volume_triplets;

        for (std::size_t time_position = 0;
             time_position < time_indices.size(); ++time_position)
        {
            const T physical_time = time_axis[time_indices[time_position]];
            T parameter_time = (physical_time - physical_lower[2]) /
                result.physical_extents[2];
            if (parameter_time < T(0) && parameter_time >
                -T(64) * std::numeric_limits<T>::epsilon())
                parameter_time = T(0);
            if (parameter_time > T(1) && parameter_time < T(1) +
                T(64) * std::numeric_limits<T>::epsilon())
                parameter_time = T(1);
            const int t_basis = registerBasis(
                parameter_time, knots[2], t_registry, result.t_bases);

            std::vector<FaceQuadrature> vertical(
                static_cast<std::size_t>(parts_x + 1) * parts_y);
            std::vector<FaceQuadrature> horizontal(
                static_cast<std::size_t>(parts_y + 1) * parts_x);
            for (int fixed = 0; fixed <= parts_x; ++fixed)
                for (int segment = 0; segment < parts_y; ++segment)
                    vertical[static_cast<std::size_t>(fixed) * parts_y +
                             segment] = buildFace(
                        0, fixed, segment, t_basis, parts_x, parts_y,
                        knots, result.physical_extents, quadrature_order,
                        x_registry, y_registry, result.x_bases,
                        result.y_bases, result.face_samples);
            for (int fixed = 0; fixed <= parts_y; ++fixed)
                for (int segment = 0; segment < parts_x; ++segment)
                    horizontal[static_cast<std::size_t>(fixed) * parts_x +
                               segment] = buildFace(
                        1, fixed, segment, t_basis, parts_y, parts_x,
                        knots, result.physical_extents, quadrature_order,
                        x_registry, y_registry, result.x_bases,
                        result.y_bases, result.face_samples);

            for (int by = 0; by < parts_y; ++by)
                for (int bx = 0; bx < parts_x; ++bx)
                {
                    const int row = static_cast<int>(time_position) * blocks +
                        by * parts_x + bx;
                    const T area = result.physical_extents[0] /
                        static_cast<T>(parts_x) * result.physical_extents[1] /
                        static_cast<T>(parts_y);
                    result.scale(row) = T(1) / std::sqrt(area);

                    const FaceQuadrature& left =
                        vertical[static_cast<std::size_t>(bx) * parts_y + by];
                    const FaceQuadrature& right = vertical[
                        static_cast<std::size_t>(bx + 1) * parts_y + by];
                    for (std::size_t index = 0; index < left.samples.size(); ++index)
                        ex_triplets.push_back(Triplet(
                            row, left.samples[index], -left.weights[index]));
                    for (std::size_t index = 0; index < right.samples.size(); ++index)
                        ex_triplets.push_back(Triplet(
                            row, right.samples[index], right.weights[index]));

                    const FaceQuadrature& bottom = horizontal[
                        static_cast<std::size_t>(by) * parts_x + bx];
                    const FaceQuadrature& top = horizontal[
                        static_cast<std::size_t>(by + 1) * parts_x + bx];
                    for (std::size_t index = 0; index < bottom.samples.size(); ++index)
                        ey_triplets.push_back(Triplet(
                            row, bottom.samples[index], -bottom.weights[index]));
                    for (std::size_t index = 0; index < top.samples.size(); ++index)
                        ey_triplets.push_back(Triplet(
                            row, top.samples[index], top.weights[index]));

                    buildVolume(
                        bx, by, t_basis, row, parts_x, parts_y, knots,
                        result.physical_extents, quadrature_order,
                        x_registry, y_registry, result.x_bases,
                        result.y_bases, result.volume_samples,
                        volume_triplets);
                }
        }

        result.Ex.resize(rows, static_cast<int>(result.face_samples.size()));
        result.Ey.resize(rows, static_cast<int>(result.face_samples.size()));
        result.V.resize(rows, static_cast<int>(result.volume_samples.size()));
        result.Ex.setFromTriplets(ex_triplets.begin(), ex_triplets.end());
        result.Ey.setFromTriplets(ey_triplets.begin(), ey_triplets.end());
        result.V.setFromTriplets(volume_triplets.begin(), volume_triplets.end());
        result.Ex.makeCompressed();
        result.Ey.makeCompressed();
        result.V.makeCompressed();

        const Vector ones = Vector::Ones(rows);
        const Vector sum_ex = result.Ex.transpose() * ones;
        const Vector sum_ey = result.Ey.transpose() * ones;
        T maximum_internal = T(0);
        for (int sample = 0;
             sample < static_cast<int>(result.face_samples.size()); ++sample)
        {
            const BurgersGLSample& face = result.face_samples[sample];
            const bool internal = face.normal_axis == 0
                ? (face.fixed_partition_index > 0 &&
                   face.fixed_partition_index < parts_x)
                : (face.fixed_partition_index > 0 &&
                   face.fixed_partition_index < parts_y);
            if (internal)
            {
                const T discrepancy = face.normal_axis == 0
                    ? std::abs(sum_ex(sample)) : std::abs(sum_ey(sample));
                maximum_internal = std::max(maximum_internal, discrepancy);
            }
        }
        result.maximum_internal_face_closure = maximum_internal;
        if (check_closure && maximum_internal > closure_tolerance)
            throw std::runtime_error(
                "Burgers integral internal-face closure check failed");

        const Vector volume_measures = result.V *
            Vector::Ones(static_cast<int>(result.volume_samples.size()));
        const T expected_area = result.physical_extents[0] /
            static_cast<T>(parts_x) * result.physical_extents[1] /
            static_cast<T>(parts_y);
        result.maximum_volume_measure_error =
            (volume_measures.array() - expected_area).abs().maxCoeff();
        const T measure_tolerance = T(512) *
            std::numeric_limits<T>::epsilon() *
            std::max<T>(T(1), expected_area);
        if (result.maximum_volume_measure_error > measure_tolerance)
            throw std::runtime_error(
                "Burgers GL volume quadrature failed the constant measure check");
        return result;
    }

    static int coefficientIndex(
        const BurgersGLBlockOperators<T>& operators,
        int x,
        int y,
        int t)
    {
        return x + operators.control_counts[0] *
            (y + operators.control_counts[1] * t);
    }

    static void evaluateFaceSample(
        const BurgersGLBlockOperators<T>& operators,
        const BurgersGLSample& sample,
        const Vector& coefficients,
        T& value,
        T& derivative_x,
        T& derivative_y)
    {
        const BurgersGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const BurgersGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const BurgersGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        value = derivative_x = derivative_y = T(0);
        for (int lt = 0; lt < 4; ++lt)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx)
                {
                    const int index = coefficientIndex(
                        operators, x.first_control + lx,
                        y.first_control + ly, t.first_control + lt);
                    const T coefficient = coefficients(index);
                    value += coefficient * x.value[lx] * y.value[ly] *
                        t.value[lt];
                    derivative_x += coefficient * x.derivative[lx] *
                        y.value[ly] * t.value[lt];
                    derivative_y += coefficient * x.value[lx] *
                        y.derivative[ly] * t.value[lt];
                }
        derivative_x /= operators.physical_extents[0];
        derivative_y /= operators.physical_extents[1];
    }

    static void evaluateVolumeSample(
        const BurgersGLBlockOperators<T>& operators,
        const BurgersGLSample& sample,
        const Vector& coefficients,
        T& value,
        T& derivative_x,
        T& derivative_y,
        T& derivative_t,
        T& second_x,
        T& second_y)
    {
        const BurgersGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const BurgersGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const BurgersGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        value = derivative_x = derivative_y = derivative_t =
            second_x = second_y = T(0);
        for (int lt = 0; lt < 4; ++lt)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx)
                {
                    const int index = coefficientIndex(
                        operators, x.first_control + lx,
                        y.first_control + ly, t.first_control + lt);
                    const T coefficient = coefficients(index);
                    value += coefficient * x.value[lx] * y.value[ly] *
                        t.value[lt];
                    derivative_x += coefficient * x.derivative[lx] *
                        y.value[ly] * t.value[lt];
                    derivative_y += coefficient * x.value[lx] *
                        y.derivative[ly] * t.value[lt];
                    derivative_t += coefficient * x.value[lx] * y.value[ly] *
                        t.derivative[lt];
                    second_x += coefficient * x.second_derivative[lx] *
                        y.value[ly] * t.value[lt];
                    second_y += coefficient * x.value[lx] *
                        y.second_derivative[ly] * t.value[lt];
                }
        derivative_x /= operators.physical_extents[0];
        derivative_y /= operators.physical_extents[1];
        derivative_t /= operators.physical_extents[2];
        second_x /= operators.physical_extents[0] *
            operators.physical_extents[0];
        second_y /= operators.physical_extents[1] *
            operators.physical_extents[1];
    }

    static void scatterFaceAdjoint(
        const BurgersGLBlockOperators<T>& operators,
        const BurgersGLSample& sample,
        T adjoint_value,
        T adjoint_derivative_x,
        T adjoint_derivative_y,
        Vector& coefficient_adjoint)
    {
        const BurgersGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const BurgersGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const BurgersGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        for (int lt = 0; lt < 4; ++lt)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx)
                {
                    const int index = coefficientIndex(
                        operators, x.first_control + lx,
                        y.first_control + ly, t.first_control + lt);
                    const T basis_value =
                        x.value[lx] * y.value[ly] * t.value[lt];
                    const T basis_dx = x.derivative[lx] * y.value[ly] *
                        t.value[lt] / operators.physical_extents[0];
                    const T basis_dy = x.value[lx] * y.derivative[ly] *
                        t.value[lt] / operators.physical_extents[1];
                    coefficient_adjoint(index) +=
                        adjoint_value * basis_value +
                        adjoint_derivative_x * basis_dx +
                        adjoint_derivative_y * basis_dy;
                }
    }

    static void scatterVolumeAdjoint(
        const BurgersGLBlockOperators<T>& operators,
        const BurgersGLSample& sample,
        T adjoint_value,
        T adjoint_derivative_x,
        T adjoint_derivative_y,
        T adjoint_derivative_t,
        Vector& coefficient_adjoint)
    {
        const BurgersGLBasis1D<T>& x = operators.x_bases[sample.x_basis];
        const BurgersGLBasis1D<T>& y = operators.y_bases[sample.y_basis];
        const BurgersGLBasis1D<T>& t = operators.t_bases[sample.t_basis];
        for (int lt = 0; lt < 4; ++lt)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx)
                {
                    const int index = coefficientIndex(
                        operators, x.first_control + lx,
                        y.first_control + ly, t.first_control + lt);
                    const T value = x.value[lx] * y.value[ly] * t.value[lt];
                    const T dx = x.derivative[lx] * y.value[ly] * t.value[lt] /
                        operators.physical_extents[0];
                    const T dy = x.value[lx] * y.derivative[ly] * t.value[lt] /
                        operators.physical_extents[1];
                    const T dt = x.value[lx] * y.value[ly] * t.derivative[lt] /
                        operators.physical_extents[2];
                    coefficient_adjoint(index) +=
                        adjoint_value * value + adjoint_derivative_x * dx +
                        adjoint_derivative_y * dy + adjoint_derivative_t * dt;
                }
    }

    static void requireCoefficients(
        const Vector& coefficients,
        int expected,
        const char* label)
    {
        if (coefficients.size() != expected || !coefficients.allFinite())
            throw std::invalid_argument(
                std::string("Invalid Burgers coefficients for ") + label);
    }
};

} // namespace mfa

#endif // MFA_BURGERS_KNOT_SPAN_INTEGRAL_BALANCE_HPP
