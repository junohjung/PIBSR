#ifndef MFA_NS_BOUNDARY_PATH_POLICY_HPP
#define MFA_NS_BOUNDARY_PATH_POLICY_HPP

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfa_ns_boundary
{

enum class Axis
{
    x,
    y
};

inline bool strictBinaryOption(
    const char* variable_name,
    const char* raw_value)
{
    if (!variable_name || !*variable_name)
        throw std::invalid_argument(
            "NS boundary option name must not be empty");
    if (!raw_value || std::string(raw_value) == "0")
        return false;
    if (std::string(raw_value) == "1")
        return true;
    throw std::invalid_argument(
        std::string(variable_name) + " must be exactly 0 or 1");
}

inline bool timeDependentEnabled(const char* raw_value)
{
    return strictBinaryOption("MFA_NS_TIME_DEPENDENT_BC", raw_value);
}

inline bool zeroPressureRequired(const char* raw_value)
{
    return strictBinaryOption("MFA_REQUIRE_ZERO_PRESSURE_BC", raw_value);
}

inline const char* axisToken(Axis axis)
{
    if (axis == Axis::x)
        return "x";
    if (axis == Axis::y)
        return "y";
    throw std::invalid_argument("invalid NS boundary axis");
}

inline std::string joinPath(
    const std::string& root,
    const std::string& relative)
{
    if (root.empty())
        throw std::invalid_argument("NS boundary data root must not be empty");
    if (relative.empty() || relative[0] == '/')
        throw std::invalid_argument(
            "NS boundary relative path must be nonempty and relative");
    if (root[root.size() - 1] == '/')
        return root + relative;
    return root + "/" + relative;
}

inline std::string legacyFilename(Axis axis, bool require_zero_pressure)
{
    if (axis == Axis::x)
        return require_zero_pressure
            ? "BCx_selection.txt"
            : "BCxvarying_ref.txt";
    if (axis == Axis::y)
        return require_zero_pressure
            ? "BCy_selection.txt"
            : "BCyvarying_ref.txt";
    throw std::invalid_argument("invalid NS boundary axis");
}

inline std::string boundaryPath(
    const std::string& data_root,
    Axis axis,
    int time_index,
    bool time_dependent,
    bool require_zero_pressure)
{
    if (time_index < 0 || time_index > 1000)
        throw std::out_of_range(
            "NS boundary time index must lie in [0, 1000]");
    if (!time_dependent)
        return joinPath(
            data_root,
            legacyFilename(axis, require_zero_pressure));

    std::ostringstream relative;
    relative << "dynamic_boundaries/BC" << axisToken(axis) << "_"
             << std::setfill('0') << std::setw(4) << time_index << ".txt";
    return joinPath(data_root, relative.str());
}

template <typename T>
std::vector<T> readStrictDynamicTable(
    const std::string& path,
    int expected_rows = 42,
    int expected_columns = 5)
{
    if (expected_rows <= 0 || expected_columns <= 0)
        throw std::invalid_argument(
            "NS dynamic boundary table dimensions must be positive");
    std::ifstream stream(path.c_str());
    if (!stream)
        throw std::runtime_error(
            "Could not open time-dependent NS boundary file '" + path + "'");

    std::vector<T> values;
    values.reserve(
        static_cast<std::size_t>(expected_rows) *
        static_cast<std::size_t>(expected_columns));
    std::string line;
    for (int row = 0; row < expected_rows; ++row)
    {
        if (!std::getline(stream, line))
            throw std::runtime_error(
                "Time-dependent NS boundary file '" + path +
                "' ended before row " + std::to_string(row));
        std::istringstream row_stream(line);
        for (int column = 0; column < expected_columns; ++column)
        {
            T value = T(0);
            if (!(row_stream >> value) ||
                !std::isfinite(static_cast<double>(value)))
                throw std::runtime_error(
                    "Malformed or nonfinite time-dependent NS boundary value in '" +
                    path + "' at row " + std::to_string(row) +
                    ", column " + std::to_string(column));
            values.push_back(value);
        }
        std::string extra;
        if (row_stream >> extra)
            throw std::runtime_error(
                "Extra token in time-dependent NS boundary file '" + path +
                "' at row " + std::to_string(row));
    }
    if (std::getline(stream, line))
        throw std::runtime_error(
            "Time-dependent NS boundary file '" + path +
            "' contains more than " + std::to_string(expected_rows) +
            " rows");
    if (values.size() !=
        static_cast<std::size_t>(expected_rows) *
        static_cast<std::size_t>(expected_columns))
        throw std::runtime_error(
            "Time-dependent NS boundary table size invariant failed");
    return values;
}

template <typename T>
T tableValue(
    const std::vector<T>& values,
    int row,
    int column,
    int columns = 5)
{
    if (row < 0 || column < 0 || columns <= 0 || column >= columns)
        throw std::out_of_range("invalid NS boundary table coordinate");
    const std::size_t index =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
        static_cast<std::size_t>(column);
    if (index >= values.size())
        throw std::out_of_range("NS boundary table coordinate exceeds payload");
    return values[index];
}

template <typename T>
void validateFrozenCoordinates(
    const std::vector<T>& values,
    Axis axis,
    int points_per_face = 21,
    int columns = 5)
{
    if (points_per_face != 21 || columns != 5 ||
        values.size() != static_cast<std::size_t>(2 * points_per_face * columns))
        throw std::invalid_argument(
            "NS dynamic boundary coordinates require the frozen 42x5 layout");
    const double tolerance = 1.0e-6;
    for (int row = 0; row < 2 * points_per_face; ++row)
    {
        double expected_x = 0.0;
        double expected_y = 0.0;
        if (axis == Axis::x)
        {
            expected_x =
                static_cast<double>(row % points_per_face) /
                static_cast<double>(points_per_face - 1);
            expected_y = row < points_per_face ? 0.0 : 1.0;
        }
        else if (axis == Axis::y)
        {
            expected_x = row % 2 == 0 ? 0.0 : 1.0;
            expected_y =
                static_cast<double>(row / 2) /
                static_cast<double>(points_per_face - 1);
        }
        else
            throw std::invalid_argument("invalid NS boundary axis");
        const double actual_x = static_cast<double>(
            tableValue(values, row, 0, columns));
        const double actual_y = static_cast<double>(
            tableValue(values, row, 1, columns));
        if (std::abs(actual_x - expected_x) > tolerance ||
            std::abs(actual_y - expected_y) > tolerance)
            throw std::runtime_error(
                "Time-dependent NS boundary coordinates do not match the "
                "frozen 21-point face layout at row " +
                std::to_string(row));
    }
}

template <typename T>
void requireZeroPressure(
    const std::vector<T>& values,
    int expected_rows = 42,
    int columns = 5,
    int pressure_column = 4)
{
    if (expected_rows <= 0 || columns <= 0 || pressure_column < 0 ||
        pressure_column >= columns ||
        values.size() !=
            static_cast<std::size_t>(expected_rows) *
            static_cast<std::size_t>(columns))
        throw std::invalid_argument(
            "invalid zero-pressure NS boundary table contract");
    for (int row = 0; row < expected_rows; ++row)
    {
        const T pressure =
            tableValue(values, row, pressure_column, columns);
        if (pressure != T(0) ||
            std::signbit(static_cast<double>(pressure)))
            throw std::runtime_error(
                "Non-positive-zero pressure appeared in the reference-blind dynamic "
                "boundary table at row " + std::to_string(row));
    }
}

} // namespace mfa_ns_boundary

#endif
