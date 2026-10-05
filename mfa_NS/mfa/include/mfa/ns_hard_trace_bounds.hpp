//--------------------------------------------------------------
// Equality bounds that preserve an open-clamped NS Stage-A trace.
//--------------------------------------------------------------

#ifndef MFA_NS_HARD_TRACE_BOUNDS_HPP
#define MFA_NS_HARD_TRACE_BOUNDS_HPP

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mfa
{

struct NSHardTraceBoundsAttestation
{
    int locked_u;
    int locked_v;
    int locked_p;
    int free_total;
    std::vector<int> locked_indices;

    NSHardTraceBoundsAttestation()
        : locked_u(0), locked_v(0), locked_p(0), free_total(0)
    {
    }
};

template <typename T>
NSHardTraceBoundsAttestation ApplyNSHardTraceBounds(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coefficients,
    const Eigen::VectorXi&                     controls,
    Eigen::Matrix<T, Eigen::Dynamic, 1>&       lower,
    Eigen::Matrix<T, Eigen::Dynamic, 1>&       upper)
{
    if (controls.size() != 3 || controls(0) < 2 || controls(1) < 2 ||
        controls(2) < 1)
        throw std::invalid_argument(
            "Hard-trace bounds require a positive three-dimensional lattice");
    const int nx = controls(0);
    const int ny = controls(1);
    const int nt = controls(2);
    const int field_size = nx * ny * nt;
    if (coefficients.size() != 3 * field_size ||
        lower.size() != coefficients.size() ||
        upper.size() != coefficients.size())
        throw std::invalid_argument(
            "Hard-trace coefficient or bound size mismatch");
    if (!coefficients.allFinite() || !lower.allFinite() || !upper.allFinite())
        throw std::invalid_argument(
            "Hard-trace coefficients and bounds must be finite");
    if ((lower.array() > upper.array()).any() ||
        (coefficients.array() < lower.array()).any() ||
        (coefficients.array() > upper.array()).any())
        throw std::invalid_argument(
            "Hard-trace input is outside the declared free bounds");

    NSHardTraceBoundsAttestation result;
    result.locked_indices.reserve(2 * field_size);
    for (int field = 0; field < 2; ++field)
        for (int time = 0; time < nt; ++time)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                {
                    if (x != 0 && x != nx - 1 && y != 0 && y != ny - 1 &&
                        time != 0)
                        continue;
                    const int local = x + nx * (y + ny * time);
                    const int global = field * field_size + local;
                    lower(global) = coefficients(global);
                    upper(global) = coefficients(global);
                    result.locked_indices.push_back(global);
                    if (field == 0)
                        ++result.locked_u;
                    else
                        ++result.locked_v;
                }
    result.free_total = coefficients.size() -
        static_cast<int>(result.locked_indices.size());
    return result;
}

template <typename T>
T NSProjectedGradientInfinity(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coefficients,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& gradient,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& lower,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& upper)
{
    if (coefficients.size() == 0 || gradient.size() != coefficients.size() ||
        lower.size() != coefficients.size() ||
        upper.size() != coefficients.size())
        throw std::invalid_argument("Projected-gradient vector size mismatch");
    if (!coefficients.allFinite() || !gradient.allFinite() ||
        !lower.allFinite() || !upper.allFinite())
        throw std::invalid_argument("Projected-gradient inputs must be finite");
    if ((lower.array() > upper.array()).any())
        throw std::invalid_argument("Projected-gradient bounds are invalid");
    return ((coefficients - gradient).cwiseMax(lower).cwiseMin(upper) -
            coefficients).cwiseAbs().maxCoeff();
}

template <typename T>
T NSHardTraceMaximumDrift(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& coefficients,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& locked_values,
    const std::vector<int>&                    locked_indices)
{
    if (coefficients.size() != locked_values.size())
        throw std::invalid_argument("Hard-trace drift vector size mismatch");
    if (!coefficients.allFinite() || !locked_values.allFinite())
        throw std::invalid_argument("Hard-trace drift vectors must be finite");
    T maximum = T(0);
    for (size_t i = 0; i < locked_indices.size(); ++i)
    {
        const int index = locked_indices[i];
        if (index < 0 || index >= coefficients.size())
            throw std::invalid_argument("Hard-trace drift index is invalid");
        const T difference =
            std::abs(coefficients(index) - locked_values(index));
        if (!std::isfinite(difference))
            throw std::invalid_argument(
                "Hard-trace drift difference must be finite");
        maximum = std::max<T>(maximum, difference);
    }
    return maximum;
}

} // namespace mfa

#endif
