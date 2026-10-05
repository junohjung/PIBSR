#include <mfa/burgers_boundary_contract.hpp>
#include <mfa/mfa.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
  using T = double;

  void fillBoundaryRows(mfa::PointSet<T>& points)
  {
    int row = 0;
    for (int time_index = 0; time_index < 201; ++time_index)
      for (int face = 0; face < 4; ++face)
        for (int face_row = 0; face_row < 41; ++face_row, ++row)
          {
            const T s = T(0.05) * T(face_row);
            points.domain(row, 0) = face == 1 ? T(0) :
                                    face == 2 ? T(2) : s;
            points.domain(row, 1) = face == 0 ? T(0) :
                                    face == 3 ? T(2) : s;
            points.domain(row, 2) = T(0.01) * T(time_index);
            points.domain(row, 3) = T(0);
            points.domain(row, 4) = T(0);
          }
    if (row != points.npts)
      throw std::runtime_error("test fixture row count mismatch");
  }

  void require(bool condition, const std::string& message)
  {
    if (!condition)
      throw std::runtime_error(message);
  }

  template <typename Function>
  void requireThrows(Function&& function, const std::string& message)
  {
    bool threw = false;
    try
      {
        function();
      }
    catch (const std::runtime_error&)
      {
        threw = true;
      }
    require(threw, message);
  }
}

int main()
{
  try
    {
      VectorXi mdims(3);
      mdims << 3, 1, 1;
      VectorX<T> lower(3);
      VectorX<T> upper(3);
      lower << T(0), T(0), T(0);
      upper << T(2), T(2), T(2);

      mfa::PointSet<T> corrected(3, mdims, 164 * 201);
      fillBoundaryRows(corrected);
      corrected.set_domain_params(lower, upper);
      const auto audit = mfa::auditBurgersBoundaryPointSet(corrected);
      require(audit.rows == 32964, "positive row audit failed");
      require(audit.face_counts ==
                std::array<int, 4>{{8241, 8241, 8241, 8241}},
              "positive face-count audit failed");
      require(audit.unique_physical_space_time_points == 32160,
              "unique-point audit failed");
      require(audit.max_parameter_error <= 1e-13,
              "positive parameter audit failed");

      VectorX<T> param(3);
      for (const int row : {0, 20, 40, 41, 61, 81, 82, 102, 122,
                            123, 143, 163, 164 * 100 + 61, 32963})
        {
          corrected.pt_params(static_cast<size_t>(row), param);
          for (int k = 0; k < 3; ++k)
            require(std::abs(param(k) - corrected.domain(row, k) / T(2)) <=
                      1e-13,
                    "representative physical-parameter map failed");
        }

      VectorXi old_grid(3);
      old_grid << 41, 4, 201;
      mfa::PointSet<T> old_structured(3, mdims, 164 * 201, old_grid);
      fillBoundaryRows(old_structured);
      old_structured.set_domain_params();
      requireThrows(
        [&]() { mfa::auditBurgersBoundaryPointSet(old_structured); },
        "former structured construction was not rejected");

      mfa::PointSet<T> corrupted(3, mdims, 164 * 201);
      fillBoundaryRows(corrupted);
      corrupted.domain(41, 0) = T(2);
      corrupted.set_domain_params(lower, upper);
      requireThrows(
        [&]() { mfa::auditBurgersBoundaryPointSet(corrupted); },
        "corrupted vertical-face row was not rejected");

      std::cout << "burgers_boundary_contract=PASS rows=" << audit.rows
                << " unique=" << audit.unique_physical_space_time_points
                << " max_parameter_error=" << audit.max_parameter_error
                << std::endl;
    }
  catch (const std::exception& error)
    {
      std::cerr << "burgers_boundary_contract=FAIL reason="
                << error.what() << std::endl;
      return 1;
    }
  return 0;
}
