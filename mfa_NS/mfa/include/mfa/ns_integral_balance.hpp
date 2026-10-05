//--------------------------------------------------------------
// Conservative block-integral balances for the 2-D incompressible
// Navier--Stokes objective.
//
// The operator acts on field values and analytic MFA derivatives that
// are already evaluated on a structured x-y-t grid.  It deliberately
// does not form a dense coefficient-space Jacobian.
//--------------------------------------------------------------

#ifndef MFA_NS_INTEGRAL_BALANCE_HPP
#define MFA_NS_INTEGRAL_BALANCE_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa
{

template <typename T>
inline T nsIntegralStrictScalar(
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
        throw std::invalid_argument(
            std::string(name) + " must be nonnegative");
    return value;
}

inline int nsIntegralStrictInteger(
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

inline bool nsIntegralStrictBoolean(const char* name, bool default_value)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        return default_value;
    if (std::string(text) == "0")
        return false;
    if (std::string(text) == "1")
        return true;
    throw std::invalid_argument(
        std::string(name) + " must be exactly 0 or 1");
}

inline std::string nsIntegralRequiredPath(const char* name)
{
    const char* text = std::getenv(name);
    if (!text || !*text)
        throw std::invalid_argument(
            std::string(name) + " is required when an integral weight is positive");
    return std::string(text);
}

inline std::vector<int> nsIntegralReadIndices(
    const std::string& path,
    const char* label)
{
    std::ifstream stream(path.c_str());
    if (!stream)
        throw std::runtime_error(
            std::string("Cannot open ") + label + ": " + path);

    std::vector<int> indices;
    std::set<int> seen;
    long long value = -1;
    while (stream >> value)
    {
        if (value < 0 || value > std::numeric_limits<int>::max())
            throw std::runtime_error(
                std::string("Out-of-range integer in ") + label);
        const int index = static_cast<int>(value);
        if (!seen.insert(index).second)
            throw std::runtime_error(
                std::string("Duplicate time index in ") + label);
        indices.push_back(index);
    }
    if (!stream.eof())
        throw std::runtime_error(
            std::string("Malformed time index in ") + label);
    if (indices.empty())
        throw std::runtime_error(
            std::string("No time indices in ") + label);
    if (!std::is_sorted(indices.begin(), indices.end()))
        throw std::runtime_error(
            std::string(label) + " must contain sorted canonical indices");
    return indices;
}

template <typename T, typename Derived>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> nsPhysicsTimeMask(
    const Eigen::MatrixBase<Derived>& times,
    T minimum_time,
    T& active_count)
{
    if (minimum_time < T(0) || !std::isfinite(minimum_time))
        throw std::invalid_argument(
            "Physics minimum time must be finite and nonnegative");
    Eigen::Matrix<T, Eigen::Dynamic, 1> mask =
        Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(times.size());
    const T tolerance =
        T(1.0e-6) * std::max<T>(T(1), std::abs(minimum_time));
    active_count = T(0);
    for (Eigen::Index row = 0; row < times.size(); ++row)
    {
        if (static_cast<T>(times(row)) + tolerance < minimum_time)
            mask(row) = T(0);
        else
            active_count += T(1);
    }
    if (active_count <= T(0))
        throw std::invalid_argument(
            "Physics minimum time excludes every row");
    return mask;
}

template <typename T>
struct NSIntegralBalanceConfig
{
    T lambda_mass;
    T lambda_xmom;
    T lambda_ymom;
    int parts_x;
    int parts_y;
    std::string quadrature;
    T physics_t_min;
    T viscosity;
    bool diagnostics;
    bool check_closure;
    T closure_tolerance;
    std::vector<int> fit_time_indices;
    std::vector<int> validation_time_indices;

    NSIntegralBalanceConfig()
        : lambda_mass(T(0)),
          lambda_xmom(T(0)),
          lambda_ymom(T(0)),
          parts_x(4),
          parts_y(4),
          quadrature("trapezoid"),
          physics_t_min(T(0)),
          viscosity(T(0.001)),
          diagnostics(false),
          check_closure(true),
          closure_tolerance(T(1.0e-12))
    {
    }

    bool enabled() const
    {
        return lambda_mass > T(0) ||
               lambda_xmom > T(0) ||
               lambda_ymom > T(0);
    }

    bool operatorsRequired() const
    {
        return enabled() || diagnostics;
    }

    static NSIntegralBalanceConfig fromEnvironment(T default_viscosity)
    {
        NSIntegralBalanceConfig config;
        config.lambda_mass =
            nsIntegralStrictScalar<T>("MFA_LAMBDA_INT_MASS", T(0));
        config.lambda_xmom =
            nsIntegralStrictScalar<T>("MFA_LAMBDA_INT_XMOM", T(0));
        config.lambda_ymom =
            nsIntegralStrictScalar<T>("MFA_LAMBDA_INT_YMOM", T(0));
        config.physics_t_min =
            nsIntegralStrictScalar<T>("MFA_NS_PHYSICS_T_MIN", T(0));
        config.viscosity =
            nsIntegralStrictScalar<T>("MFA_NS_NU", default_viscosity);
        config.diagnostics =
            nsIntegralStrictBoolean("MFA_INT_DIAGNOSTICS", false);
        if (!(config.viscosity > T(0)))
            throw std::invalid_argument("MFA_NS_NU must be positive");

        // Zero integral weights remain a strict fast path unless the caller
        // explicitly requests reference-blind diagnostic evaluation.  The
        // diagnostic mode constructs the same operators but contributes
        // exactly zero to the objective and gradient.
        if (!config.operatorsRequired())
            return config;

        config.parts_x =
            nsIntegralStrictInteger("MFA_INT_PARTS_X", 4, 1);
        config.parts_y =
            nsIntegralStrictInteger("MFA_INT_PARTS_Y", 4, 1);
        const char* quadrature = std::getenv("MFA_INT_QUADRATURE");
        config.quadrature =
            quadrature && *quadrature ? quadrature : "trapezoid";
        if (config.quadrature != "trapezoid" &&
            config.quadrature != "gl4")
            throw std::invalid_argument(
                "MFA_INT_QUADRATURE must be exactly trapezoid or gl4; "
                "GL8 is reserved for independent post-fit evaluation");
        config.check_closure =
            nsIntegralStrictBoolean("MFA_INT_CHECK_CLOSURE", true);
        config.closure_tolerance =
            nsIntegralStrictScalar<T>(
                "MFA_INT_CLOSURE_TOL", T(1.0e-12));
        if (!(config.closure_tolerance > T(0)))
            throw std::invalid_argument(
                "MFA_INT_CLOSURE_TOL must be positive");

        config.fit_time_indices = nsIntegralReadIndices(
            nsIntegralRequiredPath("MFA_INT_FIT_TIME_INDEX_FILE"),
            "MFA_INT_FIT_TIME_INDEX_FILE");
        config.validation_time_indices = nsIntegralReadIndices(
            nsIntegralRequiredPath("MFA_INT_VALID_TIME_INDEX_FILE"),
            "MFA_INT_VALID_TIME_INDEX_FILE");

        std::set<int> fit(
            config.fit_time_indices.begin(),
            config.fit_time_indices.end());
        for (size_t i = 0; i < config.validation_time_indices.size(); ++i)
            if (fit.count(config.validation_time_indices[i]))
                throw std::invalid_argument(
                    "Integral fit and validation time indices overlap");
        return config;
    }
};

template <typename T>
struct NSIntegralBlockOperators
{
    typedef Eigen::SparseMatrix<T> SparseMatrix;
    SparseMatrix V;
    SparseMatrix Ex;
    SparseMatrix Ey;
    Eigen::Matrix<T, Eigen::Dynamic, 1> scale_mass;
    Eigen::Matrix<T, Eigen::Dynamic, 1> scale_momentum;
    std::vector<int> time_indices;
    int blocks;

    NSIntegralBlockOperators() : blocks(0) {}
    int rows() const { return static_cast<int>(V.rows()); }
    size_t nonzeros() const
    {
        return static_cast<size_t>(
            V.nonZeros() + Ex.nonZeros() + Ey.nonZeros());
    }
};

template <typename T>
struct NSIntegralBalanceOperators
{
    NSIntegralBlockOperators<T> fit;
    NSIntegralBlockOperators<T> validation;
    std::vector<T> x_axis;
    std::vector<T> y_axis;
    std::vector<T> t_axis;
    std::vector<T> x_boundaries;
    std::vector<T> y_boundaries;
    int point_count;

    NSIntegralBalanceOperators() : point_count(0) {}
};

template <typename T>
struct NSIntegralFieldAdjoints
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector u;
    Vector ut;
    Vector ux;
    Vector uy;
    Vector v;
    Vector vt;
    Vector vx;
    Vector vy;
    Vector p;

    void resizeAndZero(int count)
    {
        u = Vector::Zero(count);
        ut = Vector::Zero(count);
        ux = Vector::Zero(count);
        uy = Vector::Zero(count);
        v = Vector::Zero(count);
        vt = Vector::Zero(count);
        vx = Vector::Zero(count);
        vy = Vector::Zero(count);
        p = Vector::Zero(count);
    }
};

template <typename T>
struct NSIntegralBalanceEvaluation
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    Vector raw_mass;
    Vector raw_xmom;
    Vector raw_ymom;
    T mse_mass;
    T mse_xmom;
    T mse_ymom;
    T weighted_loss;
    NSIntegralFieldAdjoints<T> adjoints;

    NSIntegralBalanceEvaluation()
        : mse_mass(T(0)),
          mse_xmom(T(0)),
          mse_ymom(T(0)),
          weighted_loss(T(0))
    {
    }
};

template <typename T>
class NSIntegralBalance
{
public:
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Triplet<T> Triplet;

    static NSIntegralBalanceOperators<T> build(
        const Matrix& coordinates,
        const NSIntegralBalanceConfig<T>& config)
    {
        if (!config.operatorsRequired())
            throw std::logic_error(
                "NSIntegralBalance::build called without objective weights "
                "or diagnostic mode");
        if (coordinates.cols() < 3 || coordinates.rows() == 0)
            throw std::invalid_argument(
                "Integral balance requires nonempty x-y-t coordinates");

        NSIntegralBalanceOperators<T> operators;
        operators.point_count = static_cast<int>(coordinates.rows());
        operators.x_axis = uniqueAxis(coordinates.col(0));
        operators.y_axis = uniqueAxis(coordinates.col(1));
        operators.t_axis = uniqueAxis(coordinates.col(2));

        // This first production operator intentionally targets the frozen
        // cavity grid.  Generalizing silently would weaken reproducibility.
        if (operators.x_axis.size() != 21 ||
            operators.y_axis.size() != 21 ||
            operators.t_axis.size() != 1001)
            throw std::invalid_argument(
                "Integral balance requires the structured 21 x 21 x 1001 cavity grid");
        requireStrictlyIncreasing(operators.x_axis, "x");
        requireStrictlyIncreasing(operators.y_axis, "y");
        requireStrictlyIncreasing(operators.t_axis, "t");
        requireFrozenAxis(operators.x_axis, T(0.05), "x");
        requireFrozenAxis(operators.y_axis, T(0.05), "y");
        requireFrozenAxis(operators.t_axis, T(0.01), "t");

        const int nx = static_cast<int>(operators.x_axis.size());
        const int ny = static_cast<int>(operators.y_axis.size());
        const int nt = static_cast<int>(operators.t_axis.size());
        if ((nx - 1) % config.parts_x != 0 ||
            (ny - 1) % config.parts_y != 0)
            throw std::invalid_argument(
                "Integral partition must divide the spatial cell counts exactly");
        if (coordinates.rows() !=
            static_cast<Eigen::Index>(nx) * ny * nt)
            throw std::invalid_argument(
                "Integral coordinates do not form a complete structured grid");

        std::vector<int> row_of(
            static_cast<size_t>(nx) * ny * nt, -1);
        const T coordinate_tolerance =
            T(128) * std::numeric_limits<T>::epsilon();
        for (int row = 0; row < coordinates.rows(); ++row)
        {
            const int ix = axisIndex(
                operators.x_axis, coordinates(row, 0), coordinate_tolerance);
            const int iy = axisIndex(
                operators.y_axis, coordinates(row, 1), coordinate_tolerance);
            const int it = axisIndex(
                operators.t_axis, coordinates(row, 2), coordinate_tolerance);
            const size_t position =
                (static_cast<size_t>(it) * ny + iy) * nx + ix;
            if (row_of[position] != -1)
                throw std::invalid_argument(
                    "Duplicate x-y-t coordinate in integral grid");
            row_of[position] = row;
        }
        if (std::find(row_of.begin(), row_of.end(), -1) != row_of.end())
            throw std::invalid_argument(
                "Missing x-y-t coordinate in integral grid");

        validateTimes(
            config.fit_time_indices, operators.t_axis,
            config.physics_t_min, "fit");
        validateTimes(
            config.validation_time_indices, operators.t_axis,
            config.physics_t_min, "validation");

        operators.fit = buildRows(
            operators.x_axis, operators.y_axis, row_of,
            nx, ny, config.parts_x, config.parts_y,
            config.fit_time_indices);
        operators.validation = buildRows(
            operators.x_axis, operators.y_axis, row_of,
            nx, ny, config.parts_x, config.parts_y,
            config.validation_time_indices);

        operators.x_boundaries.reserve(config.parts_x + 1);
        operators.y_boundaries.reserve(config.parts_y + 1);
        const int cells_x = (nx - 1) / config.parts_x;
        const int cells_y = (ny - 1) / config.parts_y;
        for (int part = 0; part <= config.parts_x; ++part)
            operators.x_boundaries.push_back(
                operators.x_axis[part * cells_x]);
        for (int part = 0; part <= config.parts_y; ++part)
            operators.y_boundaries.push_back(
                operators.y_axis[part * cells_y]);

        if (config.check_closure)
        {
            checkFaceClosure(
                operators.fit, operators.x_axis, operators.y_axis,
                row_of, nx, ny, config.parts_x, config.parts_y,
                config.closure_tolerance);
            checkFaceClosure(
                operators.validation, operators.x_axis, operators.y_axis,
                row_of, nx, ny, config.parts_x, config.parts_y,
                config.closure_tolerance);
        }
        return operators;
    }

    static NSIntegralBalanceEvaluation<T> evaluate(
        const NSIntegralBlockOperators<T>& operators,
        const Vector& u,
        const Vector& ut,
        const Vector& ux,
        const Vector& uy,
        const Vector& v,
        const Vector& vt,
        const Vector& vx,
        const Vector& vy,
        const Vector& p,
        T viscosity,
        T lambda_mass,
        T lambda_xmom,
        T lambda_ymom,
        bool compute_adjoints)
    {
        const int point_count = static_cast<int>(operators.V.cols());
        requireFieldSize(u, point_count, "u");
        requireFieldSize(ut, point_count, "ut");
        requireFieldSize(ux, point_count, "ux");
        requireFieldSize(uy, point_count, "uy");
        requireFieldSize(v, point_count, "v");
        requireFieldSize(vt, point_count, "vt");
        requireFieldSize(vx, point_count, "vx");
        requireFieldSize(vy, point_count, "vy");
        requireFieldSize(p, point_count, "p");
        if (operators.rows() <= 0)
            throw std::invalid_argument(
                "Integral evaluation requires nonempty operator rows");
        if (!(viscosity > T(0)))
            throw std::invalid_argument(
                "Integral evaluation requires positive viscosity");

        NSIntegralBalanceEvaluation<T> result;
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
            operators.V * ut +
            operators.Ex * flux_x_x +
            operators.Ey * flux_y_x;
        result.raw_ymom =
            operators.V * vt +
            operators.Ex * flux_x_y +
            operators.Ey * flux_y_y;

        const T inverse_rows = T(1) / static_cast<T>(operators.rows());
        result.mse_mass =
            (operators.scale_mass.cwiseProduct(result.raw_mass)).squaredNorm()
            * inverse_rows;
        result.mse_xmom =
            (operators.scale_momentum.cwiseProduct(result.raw_xmom)).squaredNorm()
            * inverse_rows;
        result.mse_ymom =
            (operators.scale_momentum.cwiseProduct(result.raw_ymom)).squaredNorm()
            * inverse_rows;
        result.weighted_loss =
            lambda_mass * result.mse_mass +
            lambda_xmom * result.mse_xmom +
            lambda_ymom * result.mse_ymom;

        if (!compute_adjoints)
            return result;

        result.adjoints.resizeAndZero(point_count);
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
        result.adjoints.u += ex_mass;
        result.adjoints.v += ey_mass;

        const Vector volume_x = operators.V.transpose() * z_xmom;
        const Vector ex_x = operators.Ex.transpose() * z_xmom;
        const Vector ey_x = operators.Ey.transpose() * z_xmom;
        result.adjoints.ut += volume_x;
        result.adjoints.u +=
            (T(2) * u).cwiseProduct(ex_x) +
            v.cwiseProduct(ey_x);
        result.adjoints.v += u.cwiseProduct(ey_x);
        result.adjoints.ux -= viscosity * ex_x;
        result.adjoints.uy -= viscosity * ey_x;
        result.adjoints.p += ex_x;

        const Vector volume_y = operators.V.transpose() * z_ymom;
        const Vector ex_y = operators.Ex.transpose() * z_ymom;
        const Vector ey_y = operators.Ey.transpose() * z_ymom;
        result.adjoints.vt += volume_y;
        result.adjoints.u += v.cwiseProduct(ex_y);
        result.adjoints.v +=
            u.cwiseProduct(ex_y) +
            (T(2) * v).cwiseProduct(ey_y);
        result.adjoints.vx -= viscosity * ex_y;
        result.adjoints.vy -= viscosity * ey_y;
        result.adjoints.p += ey_y;

        return result;
    }

private:
    template <typename Derived>
    static std::vector<T> uniqueAxis(
        const Eigen::MatrixBase<Derived>& column)
    {
        std::vector<T> values;
        values.reserve(static_cast<size_t>(column.size()));
        for (Eigen::Index i = 0; i < column.size(); ++i)
            values.push_back(static_cast<T>(column(i)));
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    static void requireStrictlyIncreasing(
        const std::vector<T>& axis,
        const char* name)
    {
        for (size_t i = 1; i < axis.size(); ++i)
            if (!(axis[i] > axis[i - 1]))
                throw std::invalid_argument(
                    std::string("Integral ") + name +
                    " axis is not strictly increasing");
    }

    static void requireFrozenAxis(
        const std::vector<T>& axis,
        T spacing,
        const char* name)
    {
        const T tolerance = T(1.0e-6);
        for (size_t i = 0; i < axis.size(); ++i)
        {
            const T expected = spacing * static_cast<T>(i);
            if (std::abs(axis[i] - expected) >
                tolerance * std::max<T>(T(1), std::abs(expected)))
                throw std::invalid_argument(
                    std::string("Integral frozen cavity ") + name +
                    " axis differs from its declared spacing");
        }
    }

    static int axisIndex(
        const std::vector<T>& axis,
        T value,
        T relative_tolerance)
    {
        typename std::vector<T>::const_iterator lower =
            std::lower_bound(axis.begin(), axis.end(), value);
        int best = -1;
        T error = std::numeric_limits<T>::infinity();
        if (lower != axis.end())
        {
            best = static_cast<int>(lower - axis.begin());
            error = std::abs(*lower - value);
        }
        if (lower != axis.begin())
        {
            const int candidate =
                static_cast<int>((lower - axis.begin()) - 1);
            const T candidate_error = std::abs(axis[candidate] - value);
            if (candidate_error < error)
            {
                best = candidate;
                error = candidate_error;
            }
        }
        const T scale = std::max<T>(T(1), std::abs(value));
        if (best < 0 || error > relative_tolerance * scale)
            throw std::invalid_argument(
                "Coordinate does not match a unique structured-grid axis value");
        return best;
    }

    static void validateTimes(
        const std::vector<int>& indices,
        const std::vector<T>& t_axis,
        T minimum_time,
        const char* label)
    {
        // The cavity reader obtains time from single-precision snapshot
        // metadata before storing it in T, so use a coordinate-scale
        // tolerance rather than machine epsilon here.
        const T tolerance =
            T(1.0e-6) * std::max<T>(T(1), std::abs(minimum_time));
        for (size_t i = 0; i < indices.size(); ++i)
        {
            const int index = indices[i];
            if (index < 0 || index >= static_cast<int>(t_axis.size()))
                throw std::invalid_argument(
                    std::string("Out-of-range integral ") + label +
                    " time index");
            if (t_axis[index] + tolerance < minimum_time)
                throw std::invalid_argument(
                    std::string("Integral ") + label +
                    " time index violates MFA_NS_PHYSICS_T_MIN");
        }
    }

    static std::vector<T> trapezoidWeights(
        const std::vector<T>& axis,
        int begin,
        int end)
    {
        if (begin < 0 || end >= static_cast<int>(axis.size()) || end <= begin)
            throw std::invalid_argument(
                "Invalid composite-trapezoid axis interval");
        std::vector<T> weights(static_cast<size_t>(end - begin + 1), T(0));
        for (int cell = begin; cell < end; ++cell)
        {
            const T width = axis[cell + 1] - axis[cell];
            if (!(width > T(0)))
                throw std::invalid_argument(
                    "Nonpositive composite-trapezoid cell width");
            weights[static_cast<size_t>(cell - begin)] += width / T(2);
            weights[static_cast<size_t>(cell + 1 - begin)] += width / T(2);
        }
        return weights;
    }

    static NSIntegralBlockOperators<T> buildRows(
        const std::vector<T>& x_axis,
        const std::vector<T>& y_axis,
        const std::vector<int>& row_of,
        int nx,
        int ny,
        int parts_x,
        int parts_y,
        const std::vector<int>& time_indices)
    {
        NSIntegralBlockOperators<T> result;
        const int cells_x = (nx - 1) / parts_x;
        const int cells_y = (ny - 1) / parts_y;
        const int blocks = parts_x * parts_y;
        const int rows = blocks * static_cast<int>(time_indices.size());
        const int point_count = nx * ny *
            static_cast<int>(row_of.size() / static_cast<size_t>(nx * ny));
        result.blocks = blocks;
        result.time_indices = time_indices;

        std::vector<Triplet> volume_triplets;
        std::vector<Triplet> ex_triplets;
        std::vector<Triplet> ey_triplets;
        volume_triplets.reserve(
            static_cast<size_t>(rows) *
            (cells_x + 1) * (cells_y + 1));
        ex_triplets.reserve(
            static_cast<size_t>(rows) * T(2) * (cells_y + 1));
        ey_triplets.reserve(
            static_cast<size_t>(rows) * T(2) * (cells_x + 1));

        result.scale_mass.resize(rows);
        result.scale_momentum.resize(rows);

        int output_row = 0;
        for (size_t time_position = 0;
             time_position < time_indices.size(); ++time_position)
        {
            const int it = time_indices[time_position];
            for (int by = 0; by < parts_y; ++by)
            {
                const int iy0 = by * cells_y;
                const int iy1 = (by + 1) * cells_y;
                const std::vector<T> wy =
                    trapezoidWeights(y_axis, iy0, iy1);
                for (int bx = 0; bx < parts_x; ++bx, ++output_row)
                {
                    const int ix0 = bx * cells_x;
                    const int ix1 = (bx + 1) * cells_x;
                    const std::vector<T> wx =
                        trapezoidWeights(x_axis, ix0, ix1);

                    const T area =
                        (x_axis[ix1] - x_axis[ix0]) *
                        (y_axis[iy1] - y_axis[iy0]);
                    if (!(area > T(0)))
                        throw std::invalid_argument(
                            "Integral block has nonpositive area");
                    const T length = std::sqrt(area);
                    result.scale_mass(output_row) = T(1) / length;
                    result.scale_momentum(output_row) = T(1) / length;

                    for (int iy = iy0; iy <= iy1; ++iy)
                    {
                        for (int ix = ix0; ix <= ix1; ++ix)
                        {
                            const size_t position =
                                (static_cast<size_t>(it) * ny + iy) * nx + ix;
                            volume_triplets.push_back(Triplet(
                                output_row, row_of[position],
                                wx[static_cast<size_t>(ix - ix0)] *
                                wy[static_cast<size_t>(iy - iy0)]));
                        }
                        const T weight =
                            wy[static_cast<size_t>(iy - iy0)];
                        const size_t left =
                            (static_cast<size_t>(it) * ny + iy) * nx + ix0;
                        const size_t right =
                            (static_cast<size_t>(it) * ny + iy) * nx + ix1;
                        ex_triplets.push_back(Triplet(
                            output_row, row_of[left], -weight));
                        ex_triplets.push_back(Triplet(
                            output_row, row_of[right], weight));
                    }
                    for (int ix = ix0; ix <= ix1; ++ix)
                    {
                        const T weight =
                            wx[static_cast<size_t>(ix - ix0)];
                        const size_t bottom =
                            (static_cast<size_t>(it) * ny + iy0) * nx + ix;
                        const size_t top =
                            (static_cast<size_t>(it) * ny + iy1) * nx + ix;
                        ey_triplets.push_back(Triplet(
                            output_row, row_of[bottom], -weight));
                        ey_triplets.push_back(Triplet(
                            output_row, row_of[top], weight));
                    }
                }
            }
        }

        result.V.resize(rows, point_count);
        result.Ex.resize(rows, point_count);
        result.Ey.resize(rows, point_count);
        result.V.setFromTriplets(
            volume_triplets.begin(), volume_triplets.end());
        result.Ex.setFromTriplets(
            ex_triplets.begin(), ex_triplets.end());
        result.Ey.setFromTriplets(
            ey_triplets.begin(), ey_triplets.end());
        result.V.makeCompressed();
        result.Ex.makeCompressed();
        result.Ey.makeCompressed();
        return result;
    }

    static void checkFaceClosure(
        const NSIntegralBlockOperators<T>& operators,
        const std::vector<T>& x_axis,
        const std::vector<T>& y_axis,
        const std::vector<int>& row_of,
        int nx,
        int ny,
        int parts_x,
        int parts_y,
        T tolerance)
    {
        const int blocks = parts_x * parts_y;
        const std::vector<T> full_x =
            trapezoidWeights(x_axis, 0, nx - 1);
        const std::vector<T> full_y =
            trapezoidWeights(y_axis, 0, ny - 1);
        // The spatial stencil is identical at every time.  Validate the
        // first represented slice; the structured-grid and row-uniqueness
        // checks above prove that later slices use the same topology.
        const size_t checked_times =
            std::min<size_t>(size_t(1), operators.time_indices.size());
        for (size_t time_position = 0;
             time_position < checked_times; ++time_position)
        {
            const int it = operators.time_indices[time_position];
            Vector sum_ex = Vector::Zero(operators.Ex.cols());
            Vector sum_ey = Vector::Zero(operators.Ey.cols());
            for (int block = 0; block < blocks; ++block)
            {
                const int row =
                    static_cast<int>(time_position) * blocks + block;
                sum_ex += Vector(operators.Ex.row(row).transpose());
                sum_ey += Vector(operators.Ey.row(row).transpose());
            }

            Vector expected_ex = Vector::Zero(operators.Ex.cols());
            Vector expected_ey = Vector::Zero(operators.Ey.cols());
            for (int iy = 0; iy < ny; ++iy)
            {
                const size_t left =
                    (static_cast<size_t>(it) * ny + iy) * nx;
                const size_t right = left + nx - 1;
                expected_ex(row_of[left]) = -full_y[static_cast<size_t>(iy)];
                expected_ex(row_of[right]) = full_y[static_cast<size_t>(iy)];
            }
            for (int ix = 0; ix < nx; ++ix)
            {
                const size_t bottom =
                    (static_cast<size_t>(it) * ny) * nx + ix;
                const size_t top =
                    (static_cast<size_t>(it) * ny + ny - 1) * nx + ix;
                expected_ey(row_of[bottom]) = -full_x[static_cast<size_t>(ix)];
                expected_ey(row_of[top]) = full_x[static_cast<size_t>(ix)];
            }
            if ((sum_ex - expected_ex).template lpNorm<Eigen::Infinity>() >
                    tolerance ||
                (sum_ey - expected_ey).template lpNorm<Eigen::Infinity>() >
                    tolerance)
                throw std::runtime_error(
                    "Integral internal-face closure check failed");
        }
    }

    static void requireFieldSize(
        const Vector& field,
        int expected,
        const char* name)
    {
        if (field.size() != expected)
            throw std::invalid_argument(
                std::string("Integral field-size mismatch for ") + name);
        if (!field.allFinite())
            throw std::invalid_argument(
                std::string("Non-finite integral field ") + name);
    }
};

} // namespace mfa

#endif // MFA_NS_INTEGRAL_BALANCE_HPP
