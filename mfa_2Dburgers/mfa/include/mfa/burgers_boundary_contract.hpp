#ifndef MFA_BURGERS_BOUNDARY_CONTRACT_HPP
#define MFA_BURGERS_BOUNDARY_CONTRACT_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <mfa/types.hpp>

namespace mfa
{
  template <typename T>
  struct PointSet;

  template <typename T>
  struct BurgersBoundaryAudit
  {
    int rows = 0;
    std::array<int, 4> face_counts{{0, 0, 0, 0}};
    T max_parameter_error = T(0);
    int unique_physical_space_time_points = 0;
  };

  // The released Burgers BC file contains 41 rows on each of the ordered
  // faces bottom, left, right, and top.  The same 164-row design is repeated
  // at 201 times.  It is a concatenated collection of physical points, not a
  // 41 x 4 x 201 tensor grid.
  template <typename T>
  BurgersBoundaryAudit<T> auditBurgersBoundaryPointSet(
    const PointSet<T>& points)
  {
    constexpr int rows_per_face = 41;
    constexpr int faces = 4;
    constexpr int rows_per_time = rows_per_face * faces;
    constexpr int time_count = 201;
    constexpr int total_rows = rows_per_time * time_count;

    if (points.is_structured())
      throw std::runtime_error(
        "Burgers boundary PointSet must be unstructured");
    if (points.dom_dim != 3 || points.geom_dim() != 3 ||
        points.pt_dim < 5 || points.npts != total_rows)
      throw std::runtime_error(
        "invalid Burgers boundary PointSet dimensions");

    const T parameter_tolerance = std::max(
      T(256) * std::numeric_limits<T>::epsilon(), T(1e-13));
    // BC.txt is deliberately read through the legacy float parser.  This
    // tolerance checks the frozen decimal grid without pretending the parsed
    // coordinates retain double precision.
    const T coordinate_tolerance =
      T(64) * T(std::numeric_limits<float>::epsilon());

    BurgersBoundaryAudit<T> audit;
    VectorX<T> param(points.dom_dim);
    for (int row = 0; row < points.npts; ++row)
      {
        points.pt_params(static_cast<size_t>(row), param);
        const int local_row = row % rows_per_time;
        const int time_index = row / rows_per_time;
        const int face = local_row <= 40 ? 0 :
                         local_row <= 81 ? 1 :
                         local_row <= 122 ? 2 : 3;
        const int face_row = local_row - rows_per_face * face;
        const T tangential = T(0.05) * T(face_row);
        const T expected_x = face == 1 ? T(0) :
                             face == 2 ? T(2) : tangential;
        const T expected_y = face == 0 ? T(0) :
                             face == 3 ? T(2) : tangential;
        const T expected_t = T(0.01) * T(time_index);
        const std::array<T, 3> expected_xyz{{
          expected_x, expected_y, expected_t}};

        for (int k = 0; k < points.dom_dim; ++k)
          {
            const T coordinate = points.domain(row, k);
            if (std::abs(coordinate - expected_xyz[k]) >
                coordinate_tolerance)
              throw std::runtime_error(
                "Burgers boundary rows violate the frozen face-order contract");
            const T error = std::abs(param(k) - coordinate / T(2));
            audit.max_parameter_error =
              std::max(audit.max_parameter_error, error);
          }
        if (std::abs(points.domain(row, 3)) > coordinate_tolerance ||
            std::abs(points.domain(row, 4)) > coordinate_tolerance)
          throw std::runtime_error(
            "Burgers boundary targets are not homogeneous Dirichlet data");

        ++audit.face_counts[face];
        ++audit.rows;
      }

    if (audit.face_counts !=
          std::array<int, 4>{{8241, 8241, 8241, 8241}} ||
        audit.max_parameter_error > parameter_tolerance)
      throw std::runtime_error("Burgers boundary parameter audit failed");
    audit.unique_physical_space_time_points = 160 * time_count;
    return audit;
  }
} // namespace mfa

#endif // MFA_BURGERS_BOUNDARY_CONTRACT_HPP
