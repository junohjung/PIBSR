#ifndef PIBSR_CAVITY_BLOCK_HPP
#define PIBSR_CAVITY_BLOCK_HPP

#include <mfa/block_base.hpp>
#include <mfa/mfa.hpp>
#include <mfa/types.hpp>

#include <diy/master.hpp>

#include <Eigen/Dense>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain_args.hpp"
#include "ns_boundary_path_policy.hpp"

template <typename T, typename U = T>
struct Block : public BlockBase<T, U>
{
    using Base = BlockBase<T, U>;
    using Base::bounds_maxs;
    using Base::bounds_mins;
    using Base::core_maxs;
    using Base::core_mins;
    using Base::dom_dim;
    using Base::input;
    using Base::input_BCx;
    using Base::input_BCy;
    using Base::input_IC;
    using Base::pt_dim;

    static void* create()
    {
        return mfa::create<Block>();
    }

    static void destroy(void* block)
    {
        mfa::destroy<Block>(block);
    }

    static void add(
        int gid,
        const Bounds<T>& core,
        const Bounds<T>& bounds,
        const Bounds<T>& domain,
        const RCLink<T>& link,
        diy::Master& master,
        int domain_dimension,
        int point_dimension,
        T ghost_factor = T(0))
    {
        mfa::add<Block, T>(
            gid,
            core,
            bounds,
            domain,
            link,
            master,
            domain_dimension,
            point_dimension,
            ghost_factor);
    }

    static void save(const void* block, diy::BinaryBuffer& buffer)
    {
        mfa::save<Block, T>(block, buffer);
    }

    static void load(void* block, diy::BinaryBuffer& buffer)
    {
        mfa::load<Block, T>(block, buffer);
    }

    void read_2d_structured_NSdata(
        const diy::Master::ProxyWithLink& proxy,
        mfa::MFAInfo& mfa_info,
        DomainArgs& args)
    {
        validateModel(mfa_info);

        const int grid_side = 21;
        const int points_per_snapshot = grid_side * grid_side;
        const int snapshot_count = 1001;
        const int time_column = 2;
        const float time_step = 0.01f;
        const std::string data_root = args.infile.empty() ? "." : args.infile;
        const VectorXi model_dimensions = mfa_info.model_dims();

        this->max_errs.resize(mfa_info.nvars());
        this->sum_sq_errs.resize(mfa_info.nvars());
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);

        VectorXi data_shape(3);
        data_shape << grid_side, grid_side, snapshot_count;
        input = new mfa::PointSet<T>(
            dom_dim,
            model_dimensions,
            points_per_snapshot * snapshot_count,
            data_shape);

        for (int snapshot = 0; snapshot < snapshot_count; ++snapshot)
        {
            readCoarseSnapshot(
                data_root,
                snapshot,
                points_per_snapshot,
                time_column,
                time_step,
                *input,
                snapshot * points_per_snapshot);
        }

        bounds_mins.tail(pt_dim - dom_dim) =
            input->domain.rightCols(pt_dim - dom_dim).colwise().minCoeff();
        bounds_maxs.tail(pt_dim - dom_dim) =
            input->domain.rightCols(pt_dim - dom_dim).colwise().maxCoeff();
        for (int dimension = 0; dimension < dom_dim; ++dimension)
        {
            bounds_mins(dimension) = args.min[dimension];
            bounds_maxs(dimension) = args.max[dimension];
        }
        core_mins = bounds_mins.head(dom_dim);
        core_maxs = bounds_maxs.head(dom_dim);
        input->set_domain_params();

        VectorXi initial_shape(3);
        initial_shape << grid_side, grid_side, 1;
        input_IC = new mfa::PointSet<T>(
            dom_dim,
            model_dimensions,
            points_per_snapshot,
            initial_shape);
        readCoarseSnapshot(
            data_root,
            0,
            points_per_snapshot,
            time_column,
            time_step,
            *input_IC,
            0);

        bounds_mins(0) = 0.0;
        bounds_maxs(0) = 1.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(1) = 1.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(2) = 0.0;
        bounds_mins.tail(pt_dim - dom_dim) =
            input_IC->domain.rightCols(pt_dim - dom_dim).colwise().minCoeff();
        bounds_maxs.tail(pt_dim - dom_dim) =
            input_IC->domain.rightCols(pt_dim - dom_dim).colwise().maxCoeff();
        input_IC->set_domain_params();

        const bool require_zero_pressure =
            mfa_ns_boundary::zeroPressureRequired(
                std::getenv("MFA_REQUIRE_ZERO_PRESSURE_BC"));
        const bool time_dependent_boundary =
            mfa_ns_boundary::timeDependentEnabled(
                std::getenv("MFA_NS_TIME_DEPENDENT_BC"));

        input_BCx = readBoundarySeries(
            data_root,
            mfa_ns_boundary::Axis::x,
            time_dependent_boundary,
            require_zero_pressure,
            model_dimensions,
            snapshot_count,
            time_column,
            time_step);
        input_BCy = readBoundarySeries(
            data_root,
            mfa_ns_boundary::Axis::y,
            time_dependent_boundary,
            require_zero_pressure,
            model_dimensions,
            snapshot_count,
            time_column,
            time_step);

        bounds_mins(0) = 0.0;
        bounds_maxs(0) = 1.0;
        bounds_mins(1) = 0.0;
        bounds_maxs(1) = 1.0;
        bounds_mins(2) = 0.0;
        bounds_maxs(2) = 10.0;

        if (require_zero_pressure)
        {
            std::cerr
                << "PIMFA_NS_BOUNDARY_CONFIG"
                << " zero_pressure_required=1"
                << " x_rows_per_time=42"
                << " y_rows_per_time=42"
                << " time_count=" << snapshot_count << '\n';
        }

        this->setup_MFA(proxy, mfa_info);
    }

  private:
    static void validateModel(const mfa::MFAInfo& mfa_info)
    {
        if (mfa_info.dom_dim != 3 ||
            mfa_info.pt_dim() != 6 ||
            mfa_info.nvars() != 3 ||
            mfa_info.geom_dim() != 3 ||
            mfa_info.var_dim(0) != 1 ||
            mfa_info.var_dim(1) != 1 ||
            mfa_info.var_dim(2) != 1)
        {
            throw std::invalid_argument(
                "NS2D requires a 3D domain with scalar u, v, and p fields");
        }
    }

    static std::string coarseSnapshotPath(
        const std::string& data_root,
        int snapshot)
    {
        std::ostringstream path;
        path << data_root << "/snapshots/coarse_"
             << std::setfill('0') << std::setw(4) << snapshot << ".txt";
        return path.str();
    }

    static void readCoarseSnapshot(
        const std::string& data_root,
        int snapshot,
        int row_count,
        int time_column,
        float time_step,
        mfa::PointSet<T>& destination,
        int row_offset)
    {
        const std::string path = coarseSnapshotPath(data_root, snapshot);
        FILE* stream = std::fopen(path.c_str(), "r");
        if (!stream)
            throw std::runtime_error("Could not read snapshot file '" + path + "'");

        for (int row = 0; row < row_count; ++row)
        {
            for (int column = 0; column < destination.pt_dim; ++column)
            {
                if (column == time_column)
                {
                    destination.domain(row_offset + row, column) =
                        time_step * snapshot;
                    continue;
                }

                float value = 0.0f;
                if (std::fscanf(stream, "%f", &value) != 1 ||
                    !std::isfinite(static_cast<double>(value)))
                {
                    std::fclose(stream);
                    std::ostringstream message;
                    message << "Malformed snapshot file '" << path
                            << "' at row " << row
                            << ", stored column " << column;
                    throw std::runtime_error(message.str());
                }
                destination.domain(row_offset + row, column) = value;
            }
        }

        closeAtExpectedEnd(stream, path, "snapshot");
    }

    static std::vector<float> readLegacyBoundaryTable(
        const std::string& path,
        int rows,
        int columns)
    {
        FILE* stream = std::fopen(path.c_str(), "r");
        if (!stream)
            throw std::runtime_error("Could not read boundary file '" + path + "'");

        std::vector<float> values;
        values.reserve(
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(columns));
        for (int row = 0; row < rows; ++row)
        {
            for (int column = 0; column < columns; ++column)
            {
                float value = 0.0f;
                if (std::fscanf(stream, "%f", &value) != 1 ||
                    !std::isfinite(static_cast<double>(value)))
                {
                    std::fclose(stream);
                    std::ostringstream message;
                    message << "Malformed boundary file '" << path
                            << "' at row " << row
                            << ", column " << column;
                    throw std::runtime_error(message.str());
                }
                values.push_back(value);
            }
        }

        closeAtExpectedEnd(stream, path, "boundary");
        return values;
    }

    static void closeAtExpectedEnd(
        FILE* stream,
        const std::string& path,
        const char* input_kind)
    {
        int character = std::fgetc(stream);
        while (character != EOF &&
               std::isspace(static_cast<unsigned char>(character)))
        {
            character = std::fgetc(stream);
        }

        const bool read_failed = std::ferror(stream) != 0;
        const int close_status = std::fclose(stream);
        if (character != EOF)
            throw std::runtime_error(
                "Unexpected trailing token in " + std::string(input_kind) +
                " file '" + path + "'");
        if (read_failed)
            throw std::runtime_error(
                "Could not finish reading " + std::string(input_kind) +
                " file '" + path + "'");
        if (close_status != 0)
            throw std::runtime_error(
                "Could not close " + std::string(input_kind) +
                " file '" + path + "'");
    }

    mfa::PointSet<T>* readBoundarySeries(
        const std::string& data_root,
        mfa_ns_boundary::Axis axis,
        bool time_dependent,
        bool require_zero_pressure,
        const VectorXi& model_dimensions,
        int snapshot_count,
        int time_column,
        float time_step)
    {
        const int rows_per_snapshot = 42;
        const int file_columns = 5;
        VectorXi boundary_shape(3);
        if (axis == mfa_ns_boundary::Axis::x)
            boundary_shape << 21, 2, snapshot_count;
        else
            boundary_shape << 2, 21, snapshot_count;

        std::unique_ptr<mfa::PointSet<T>> boundary(
            new mfa::PointSet<T>(
                dom_dim,
                model_dimensions,
                rows_per_snapshot * snapshot_count,
                boundary_shape));

        const std::string axis_name =
            mfa_ns_boundary::axisToken(axis);
        if (time_dependent)
        {
            std::cerr << "NS boundary policy: reading one 42-row "
                      << axis_name
                      << "-boundary file per represented time from "
                      << "dynamic_boundaries/BC" << axis_name
                      << "_####.txt.\n";
        }
        else
        {
            std::cerr << "NS boundary policy: the 42 " << axis_name
                      << "-boundary rows in "
                      << mfa_ns_boundary::legacyFilename(
                             axis,
                             require_zero_pressure)
                      << " are repeated at each represented time.\n";
        }

        for (int snapshot = 0; snapshot < snapshot_count; ++snapshot)
        {
            const std::string path = mfa_ns_boundary::boundaryPath(
                data_root,
                axis,
                snapshot,
                time_dependent,
                require_zero_pressure);
            std::vector<float> values;
            if (time_dependent)
            {
                values = mfa_ns_boundary::readStrictDynamicTable<float>(path);
                mfa_ns_boundary::validateFrozenCoordinates(values, axis);
            }
            else
            {
                values = readLegacyBoundaryTable(
                    path,
                    rows_per_snapshot,
                    file_columns);
            }
            if (require_zero_pressure)
                mfa_ns_boundary::requireZeroPressure(values);

            for (int row = 0; row < rows_per_snapshot; ++row)
            {
                const int destination_row =
                    snapshot * rows_per_snapshot + row;
                for (int column = 0; column < boundary->pt_dim; ++column)
                {
                    if (column == time_column)
                    {
                        boundary->domain(destination_row, column) =
                            time_step * snapshot;
                        continue;
                    }
                    const int file_column =
                        column < time_column ? column : column - 1;
                    boundary->domain(destination_row, column) =
                        mfa_ns_boundary::tableValue(
                            values,
                            row,
                            file_column,
                            file_columns);
                }
            }
        }

        boundary->set_domain_params();
        return boundary.release();
    }
};

#endif
