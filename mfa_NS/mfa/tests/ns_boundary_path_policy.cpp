#include "ns_boundary_path_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Exception>
void requireThrows(
    const std::function<void()>& function,
    const std::string& message)
{
    try
    {
        function();
    }
    catch (const Exception&)
    {
        return;
    }
    throw std::runtime_error(message);
}

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        char pattern[] = "/tmp/pimfa-ns-boundary-policy-XXXXXX";
        char* created = ::mkdtemp(pattern);
        if (!created)
            throw std::runtime_error("could not create boundary-policy test directory");
        root_ = created;
    }

    ~TemporaryDirectory()
    {
        for (std::size_t index = 0; index < files_.size(); ++index)
            std::remove(files_[index].c_str());
        ::rmdir(root_.c_str());
    }

    std::string path(const std::string& filename)
    {
        const std::string result = root_ + "/" + filename;
        files_.push_back(result);
        return result;
    }

private:
    std::string root_;
    std::vector<std::string> files_;
};

enum class Mutation
{
    none,
    truncate,
    extra_token,
    extra_row,
    nonfinite,
    bad_coordinate,
    nonzero_pressure,
    negative_zero_pressure
};

void writeTable(
    const std::string& path,
    mfa_ns_boundary::Axis axis,
    Mutation mutation)
{
    std::ofstream stream(path.c_str());
    if (!stream)
        throw std::runtime_error("could not create boundary-policy test input");
    const int rows = mutation == Mutation::truncate ? 41 : 42;
    for (int row = 0; row < rows; ++row)
    {
        double x = 0.0;
        double y = 0.0;
        if (axis == mfa_ns_boundary::Axis::x)
        {
            x = static_cast<double>(row % 21) / 20.0;
            y = row < 21 ? 0.0 : 1.0;
        }
        else
        {
            x = row % 2 == 0 ? 0.0 : 1.0;
            y = static_cast<double>(row / 2) / 20.0;
        }
        if (mutation == Mutation::bad_coordinate && row == 7)
            x += 0.01;
        const double u = 0.1 * row;
        const double v = -0.2 * row;
        const double pressure =
            mutation == Mutation::nonzero_pressure && row == 9 ? 0.5 : 0.0;
        stream << x << " " << y << " ";
        if (mutation == Mutation::nonfinite && row == 8)
            stream << "nan";
        else
            stream << u;
        stream << " " << v << " ";
        if (mutation == Mutation::negative_zero_pressure && row == 9)
            stream << "-0";
        else
            stream << pressure;
        if (mutation == Mutation::extra_token && row == 6)
            stream << " 99";
        stream << "\n";
    }
    if (mutation == Mutation::extra_row)
        stream << "0 0 0 0 0\n";
}

void testEnvironmentPolicy()
{
    const char* malformed[] = {"", "true", "01", " 1", "2", "-1"};
    ::unsetenv("MFA_NS_TIME_DEPENDENT_BC");

    ::unsetenv("MFA_REQUIRE_ZERO_PRESSURE_BC");
    require(
        !mfa_ns_boundary::zeroPressureRequired(
            std::getenv("MFA_REQUIRE_ZERO_PRESSURE_BC")),
        "unset zero-pressure policy unexpectedly enabled the requirement");
    ::setenv("MFA_REQUIRE_ZERO_PRESSURE_BC", "0", 1);
    require(
        !mfa_ns_boundary::zeroPressureRequired(
            std::getenv("MFA_REQUIRE_ZERO_PRESSURE_BC")),
        "explicit zero unexpectedly enabled the zero-pressure requirement");
    ::setenv("MFA_REQUIRE_ZERO_PRESSURE_BC", "1", 1);
    require(
        mfa_ns_boundary::zeroPressureRequired(
            std::getenv("MFA_REQUIRE_ZERO_PRESSURE_BC")),
        "explicit one did not enable the zero-pressure requirement");
    for (std::size_t index = 0;
         index < sizeof(malformed) / sizeof(malformed[0]);
         ++index)
        requireThrows<std::invalid_argument>(
            std::bind(
                mfa_ns_boundary::zeroPressureRequired,
                malformed[index]),
            std::string("malformed zero-pressure option was accepted: '") +
                malformed[index] + "'");
    ::unsetenv("MFA_REQUIRE_ZERO_PRESSURE_BC");
    require(
        !mfa_ns_boundary::timeDependentEnabled(
            std::getenv("MFA_NS_TIME_DEPENDENT_BC")),
        "unset opt-in environment unexpectedly enabled dynamic boundaries");
    ::setenv("MFA_NS_TIME_DEPENDENT_BC", "0", 1);
    require(
        !mfa_ns_boundary::timeDependentEnabled(
            std::getenv("MFA_NS_TIME_DEPENDENT_BC")),
        "explicit zero unexpectedly enabled dynamic boundaries");
    ::setenv("MFA_NS_TIME_DEPENDENT_BC", "1", 1);
    require(
        mfa_ns_boundary::timeDependentEnabled(
            std::getenv("MFA_NS_TIME_DEPENDENT_BC")),
        "explicit one did not enable dynamic boundaries");
    for (std::size_t index = 0;
         index < sizeof(malformed) / sizeof(malformed[0]);
         ++index)
        requireThrows<std::invalid_argument>(
            std::bind(
                mfa_ns_boundary::timeDependentEnabled,
                malformed[index]),
            std::string("malformed opt-in value was accepted: '") +
                malformed[index] + "'");
    ::unsetenv("MFA_NS_TIME_DEPENDENT_BC");
}

void testPathPolicy()
{
    using mfa_ns_boundary::Axis;
    require(
        mfa_ns_boundary::boundaryPath("/data", Axis::x, 0, false, false) ==
            "/data/BCxvarying_ref.txt",
        "legacy x-boundary path changed");
    require(
        mfa_ns_boundary::boundaryPath("/data/", Axis::y, 999, false, true) ==
            "/data/BCy_selection.txt",
        "legacy zero-pressure y-boundary path changed");
    require(
        mfa_ns_boundary::boundaryPath("/data", Axis::x, 0, true, true) ==
            "/data/dynamic_boundaries/BCx_0000.txt",
        "dynamic x-boundary path for t0 is wrong");
    require(
        mfa_ns_boundary::boundaryPath("/data", Axis::y, 7, true, false) ==
            "/data/dynamic_boundaries/BCy_0007.txt",
        "dynamic y-boundary path did not zero-pad the index");
    require(
        mfa_ns_boundary::boundaryPath("/data", Axis::x, 1000, true, false) ==
            "/data/dynamic_boundaries/BCx_1000.txt",
        "dynamic x-boundary path for final time is wrong");
    requireThrows<std::out_of_range>(
        std::bind(
            mfa_ns_boundary::boundaryPath,
            "/data",
            Axis::x,
            -1,
            true,
            false),
        "negative dynamic time index was accepted");
    requireThrows<std::out_of_range>(
        std::bind(
            mfa_ns_boundary::boundaryPath,
            "/data",
            Axis::x,
            1001,
            true,
            false),
        "out-of-range dynamic time index was accepted");
    requireThrows<std::invalid_argument>(
        std::bind(
            mfa_ns_boundary::boundaryPath,
            "",
            Axis::x,
            0,
            true,
            false),
        "empty data root was accepted");
    requireThrows<std::invalid_argument>(
        std::bind(
            mfa_ns_boundary::boundaryPath,
            "/data",
            static_cast<Axis>(99),
            0,
            true,
            false),
        "invalid boundary axis was accepted");
}

void testStrictTables()
{
    using mfa_ns_boundary::Axis;
    TemporaryDirectory temporary;

    const std::string valid_x = temporary.path("valid_x.txt");
    writeTable(valid_x, Axis::x, Mutation::none);
    const std::vector<double> x_values =
        mfa_ns_boundary::readStrictDynamicTable<double>(valid_x);
    require(x_values.size() == 42u * 5u, "valid x table has wrong payload size");
    mfa_ns_boundary::validateFrozenCoordinates(x_values, Axis::x);
    mfa_ns_boundary::requireZeroPressure(x_values);
    require(
        mfa_ns_boundary::tableValue(x_values, 10, 0) == 0.5,
        "valid x table value changed");

    const std::string valid_y = temporary.path("valid_y.txt");
    writeTable(valid_y, Axis::y, Mutation::none);
    const std::vector<double> y_values =
        mfa_ns_boundary::readStrictDynamicTable<double>(valid_y);
    mfa_ns_boundary::validateFrozenCoordinates(y_values, Axis::y);
    mfa_ns_boundary::requireZeroPressure(y_values);

    const std::string missing = temporary.path("missing.txt");
    requireThrows<std::runtime_error>(
        std::bind(
            mfa_ns_boundary::readStrictDynamicTable<double>,
            missing,
            42,
            5),
        "missing dynamic boundary file was accepted");

    const Mutation invalid_mutations[] = {
        Mutation::truncate,
        Mutation::extra_token,
        Mutation::extra_row,
        Mutation::nonfinite
    };
    for (std::size_t index = 0;
         index < sizeof(invalid_mutations) / sizeof(invalid_mutations[0]);
         ++index)
    {
        const std::string path =
            temporary.path("malformed_" + std::to_string(index) + ".txt");
        writeTable(path, Axis::x, invalid_mutations[index]);
        requireThrows<std::runtime_error>(
            std::bind(
                mfa_ns_boundary::readStrictDynamicTable<double>,
                path,
                42,
                5),
            "malformed dynamic boundary table was accepted");
    }

    const std::string bad_coordinate = temporary.path("bad_coordinate.txt");
    writeTable(bad_coordinate, Axis::x, Mutation::bad_coordinate);
    const std::vector<double> bad_coordinate_values =
        mfa_ns_boundary::readStrictDynamicTable<double>(bad_coordinate);
    requireThrows<std::runtime_error>(
        std::bind(
            mfa_ns_boundary::validateFrozenCoordinates<double>,
            bad_coordinate_values,
            Axis::x,
            21,
            5),
        "incorrect dynamic boundary coordinates were accepted");

    const std::string nonzero_pressure = temporary.path("nonzero_pressure.txt");
    writeTable(nonzero_pressure, Axis::x, Mutation::nonzero_pressure);
    const std::vector<double> nonzero_pressure_values =
        mfa_ns_boundary::readStrictDynamicTable<double>(nonzero_pressure);
    mfa_ns_boundary::validateFrozenCoordinates(nonzero_pressure_values, Axis::x);
    requireThrows<std::runtime_error>(
        std::bind(
            mfa_ns_boundary::requireZeroPressure<double>,
            nonzero_pressure_values,
            42,
            5,
            4),
        "nonzero dynamic pressure was accepted under the zero-pressure policy");

    const std::string negative_zero_pressure =
        temporary.path("negative_zero_pressure.txt");
    writeTable(
        negative_zero_pressure,
        Axis::x,
        Mutation::negative_zero_pressure);
    const std::vector<double> negative_zero_pressure_values =
        mfa_ns_boundary::readStrictDynamicTable<double>(negative_zero_pressure);
    requireThrows<std::runtime_error>(
        std::bind(
            mfa_ns_boundary::requireZeroPressure<double>,
            negative_zero_pressure_values,
            42,
            5,
            4),
        "negative-zero dynamic pressure was accepted under the +0 policy");

    requireThrows<std::out_of_range>(
        std::bind(
            mfa_ns_boundary::tableValue<double>,
            x_values,
            42,
            0,
            5),
        "out-of-range table coordinate was accepted");
}

} // namespace

int main()
{
    try
    {
        testEnvironmentPolicy();
        testPathPolicy();
        testStrictTables();
        std::cout
            << "PASS NS time-dependent boundary path and strict-table policy\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
