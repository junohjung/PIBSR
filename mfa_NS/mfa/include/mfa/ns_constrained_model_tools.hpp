//--------------------------------------------------------------
// Read-only model utilities for the prospective cavity V2 method.
//
// This header is opt-in and does not alter the historical flowdata path.
//--------------------------------------------------------------

#ifndef MFA_NS_CONSTRAINED_MODEL_TOOLS_HPP
#define MFA_NS_CONSTRAINED_MODEL_TOOLS_HPP

#include <mfa/ns_constrained_minimax.hpp>

#include <Eigen/Core>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace mfa
{

inline void nsConstrainedRequireRegularNonSymlink(
    const std::string& path,
    const char* role)
{
    struct stat status;
    if (::lstat(path.c_str(), &status) != 0)
        throw std::runtime_error(
            std::string("Cannot inspect constrained V2 ") + role +
            ": " + path + ": " + std::strerror(errno));
    if (S_ISLNK(status.st_mode) || !S_ISREG(status.st_mode))
        throw std::runtime_error(
            std::string("Constrained V2 ") + role +
            " must be a regular non-symlink file: " + path);
}

inline void nsConstrainedRequireAbsentPath(
    const std::string& path,
    const char* role)
{
    struct stat status;
    if (::lstat(path.c_str(), &status) == 0)
        throw std::runtime_error(
            std::string("Constrained V2 ") + role +
            " must not already exist: " + path);
    if (errno != ENOENT)
        throw std::runtime_error(
            std::string("Cannot inspect constrained V2 ") + role +
            ": " + path + ": " + std::strerror(errno));
}

inline void nsConstrainedWriteNewFile(
    const std::string& path,
    const std::string& payload)
{
    const int descriptor = ::open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (descriptor < 0)
        throw std::runtime_error(
            "Cannot create constrained V2 output exclusively: " + path +
            ": " + std::strerror(errno));
    bool complete = false;
    try
    {
        std::size_t offset = 0;
        while (offset < payload.size())
        {
            const ssize_t count = ::write(
                descriptor,
                payload.data() + offset,
                payload.size() - offset);
            if (count < 0)
            {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error(
                    "Failed writing constrained V2 output: " + path +
                    ": " + std::strerror(errno));
            }
            if (count == 0)
                throw std::runtime_error(
                    "Short write to constrained V2 output: " + path);
            offset += static_cast<std::size_t>(count);
        }
        if (::fsync(descriptor) != 0)
            throw std::runtime_error(
                "Failed syncing constrained V2 output: " + path +
                ": " + std::strerror(errno));
        if (::close(descriptor) != 0)
            throw std::runtime_error(
                "Failed closing constrained V2 output: " + path +
                ": " + std::strerror(errno));
        complete = true;
    }
    catch (...)
    {
        if (!complete)
        {
            ::close(descriptor);
            ::unlink(path.c_str());
        }
        throw;
    }
}

template <typename T, typename MFAModel>
inline Eigen::Matrix<T, Eigen::Dynamic, 1>
nsConstrainedExtractFrozenModelCoefficients(const MFAModel& model)
{
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    const Eigen::Vector3i frozen_degree(3, 3, 3);
    const Eigen::Vector3i frozen_lattice(20, 20, 50);
    const Eigen::Index per_field = 20000;
    if (model.dom_dim != 3 || model.nvars() != 3)
        throw std::runtime_error(
            "Constrained V2 model must contain exactly three ordered "
            "science variables u,v,p in a three-dimensional domain");

    Vector coefficients(3 * per_field);
    for (int field = 0; field < 3; ++field)
    {
        const auto& variable = model.var(field);
        if (variable.dom_dim != 3 ||
            variable.dim() != 1 ||
            variable.p.size() != 3 ||
            variable.p != frozen_degree ||
            variable.tmesh.tensor_prods.size() != 1)
            throw std::runtime_error(
                "Constrained V2 model variable " +
                std::to_string(field) +
                " violates the frozen scalar degree-three contract");
        const auto& tensor = variable.tmesh.tensor_prods[0];
        if (tensor.nctrl_pts.size() != 3 ||
            tensor.nctrl_pts != frozen_lattice ||
            tensor.ctrl_pts.rows() != per_field ||
            tensor.ctrl_pts.cols() != 1)
            throw std::runtime_error(
                "Constrained V2 model variable " +
                std::to_string(field) +
                " violates the frozen 20 x 20 x 50 scalar lattice");
        if (!tensor.ctrl_pts.allFinite())
            throw std::runtime_error(
                "Constrained V2 model contains nonfinite control points");
        if (tensor.weights.size() != per_field ||
            !tensor.weights.allFinite() ||
            !(tensor.weights.array() == T(1)).all())
            throw std::runtime_error(
                "Constrained V2 model rational weights are not exact ones");
        if (variable.tmesh.dom_dim_ != 3 ||
            variable.tmesh.p_ != frozen_degree ||
            variable.tmesh.min_dim_ != variable.min_dim ||
            variable.tmesh.max_dim_ != variable.max_dim ||
            variable.tmesh.max_level != 0 ||
            variable.tmesh.all_knots.size() != 3 ||
            variable.tmesh.all_knot_levels.size() != 3 ||
            variable.tmesh.all_knot_param_idxs.size() != 3 ||
            tensor.level != 0 ||
            tensor.knot_mins.size() != 3 ||
            tensor.knot_maxs.size() != 3 ||
            tensor.knot_idxs.size() != 3)
            throw std::runtime_error(
                "Constrained V2 model t-mesh metadata violates the "
                "single full-span level-zero basis contract");
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            const std::size_t expected_knot_count =
                static_cast<std::size_t>(
                    frozen_lattice(dimension) +
                    frozen_degree(dimension) + 1);
            if (variable.tmesh.all_knots[dimension].size() !=
                    expected_knot_count ||
                variable.tmesh.all_knot_levels[dimension].size() !=
                    expected_knot_count ||
                tensor.knot_mins[dimension] != 0 ||
                tensor.knot_maxs[dimension] !=
                    expected_knot_count - 1 ||
                tensor.knot_idxs[dimension].size() !=
                    expected_knot_count)
                throw std::runtime_error(
                    "Constrained V2 model does not contain the exact "
                    "full-span tensor knot topology");
            if ((!tensor.next.empty() &&
                 (tensor.next.size() != 3 ||
                  !tensor.next[dimension].empty())) ||
                (!tensor.prev.empty() &&
                 (tensor.prev.size() != 3 ||
                  !tensor.prev[dimension].empty())))
                throw std::runtime_error(
                    "Constrained V2 single-tensor model contains a "
                    "neighbor topology");
            std::vector<T> expected_knots(expected_knot_count, T(0));
            const int degree = frozen_degree(dimension);
            const int control_count = frozen_lattice(dimension);
            for (int i = 0; i < degree + 1; ++i)
                expected_knots[expected_knot_count - 1 - i] = T(1);
            const T step =
                T(1) / static_cast<T>(control_count - degree);
            for (int j = 1; j <= control_count - degree - 1; ++j)
                expected_knots[degree + j] =
                    expected_knots[degree + j - 1] + step;
            for (std::size_t knot = 0;
                 knot < expected_knot_count;
                 ++knot)
                if (!std::isfinite(
                        variable.tmesh.all_knots[dimension][knot]) ||
                    variable.tmesh.all_knots[dimension][knot] !=
                        expected_knots[knot] ||
                    variable.tmesh.all_knot_levels[dimension][knot] != 0 ||
                    tensor.knot_idxs[dimension][knot] != knot)
                    throw std::runtime_error(
                        "Constrained V2 model knot values/levels are not "
                        "the exact frozen open-uniform basis");
        }
        coefficients.segment(field * per_field, per_field) =
            tensor.ctrl_pts.col(0);
    }
    return coefficients;
}

} // namespace mfa

#endif
