#ifndef PIBSR_BURGERS_BLOCK_HPP
#define PIBSR_BURGERS_BLOCK_HPP

#include <mfa/block_base.hpp>
#include <mfa/burgers_boundary_contract.hpp>

#include <diy/master.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T, typename U = T>
struct Block : public BlockBase<T, U>
{
    typedef BlockBase<T, U> Base;
    // The released reader parsed these decimal files through float.  Preserve
    // that conversion before storing values in the double-precision PointSet.
    typedef std::array<float, 4> SpatialRow;

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

    void read_burgers_data(
        const diy::Master::ProxyWithLink& proxy,
        mfa::MFAInfo& mfa_info,
        const std::string& data_directory)
    {
        const int domain_dimension = 3;
        const int point_dimension = 5;
        const int grid_width = 41;
        const int spatial_rows = grid_width * grid_width;
        const int snapshot_count = 201;
        const int boundary_rows = 164;
        const float time_step = 0.01f;

        assert(mfa_info.dom_dim == domain_dimension);
        assert(mfa_info.pt_dim() == point_dimension);
        assert(mfa_info.nvars() == 2);
        assert(mfa_info.geom_dim() == domain_dimension);
        assert(mfa_info.var_dim(0) == 1);
        assert(mfa_info.var_dim(1) == 1);

        const VectorXi model_dimensions = mfa_info.model_dims();
        this->max_errs.resize(mfa_info.nvars());
        this->sum_sq_errs.resize(mfa_info.nvars());
        this->core_mins.resize(domain_dimension);
        this->core_maxs.resize(domain_dimension);
        this->bounds_mins.resize(point_dimension);
        this->bounds_maxs.resize(point_dimension);

        VectorXi grid_shape(domain_dimension);
        grid_shape << grid_width, grid_width, snapshot_count;
        this->input = new mfa::PointSet<T>(
            domain_dimension,
            model_dimensions,
            spatial_rows * snapshot_count,
            grid_shape);

        std::string observed_directory = join_path(data_directory, "observed");
        const std::string first_snapshot = snapshot_path(observed_directory, 0);
        std::ifstream snapshot_probe(first_snapshot.c_str());
        if (!snapshot_probe)
            observed_directory = data_directory;

        for (int snapshot = 0; snapshot < snapshot_count; ++snapshot)
        {
            const std::vector<SpatialRow> rows = read_rows(
                snapshot_path(observed_directory, snapshot),
                spatial_rows);
            copy_rows(
                *this->input,
                rows,
                snapshot * spatial_rows,
                static_cast<T>(time_step * snapshot));
        }

        this->bounds_mins.tail(point_dimension - domain_dimension) =
            this->input->domain.rightCols(point_dimension - domain_dimension)
                .colwise().minCoeff();
        this->bounds_maxs.tail(point_dimension - domain_dimension) =
            this->input->domain.rightCols(point_dimension - domain_dimension)
                .colwise().maxCoeff();
        set_physical_bounds(T(0), T(2));
        this->core_mins = this->bounds_mins.head(domain_dimension);
        this->core_maxs = this->bounds_maxs.head(domain_dimension);
        this->input->set_domain_params();

        VectorXi initial_shape(domain_dimension);
        initial_shape << grid_width, grid_width, 1;
        this->input_IC = new mfa::PointSet<T>(
            domain_dimension,
            model_dimensions,
            spatial_rows,
            initial_shape);
        copy_rows(
            *this->input_IC,
            read_rows(join_path(observed_directory, "IC.txt"), spatial_rows),
            0,
            T(0));

        set_physical_bounds(T(0), T(2));
        this->bounds_mins.tail(point_dimension - domain_dimension) =
            this->input_IC->domain.rightCols(point_dimension - domain_dimension)
                .colwise().minCoeff();
        this->bounds_maxs.tail(point_dimension - domain_dimension) =
            this->input_IC->domain.rightCols(point_dimension - domain_dimension)
                .colwise().maxCoeff();
        this->input_IC->set_domain_params();

        const std::vector<SpatialRow> boundary = read_rows(
            join_path(data_directory, "BC.txt"),
            boundary_rows);
        this->input_BC = new mfa::PointSet<T>(
            domain_dimension,
            model_dimensions,
            boundary_rows * snapshot_count);
        for (int snapshot = 0; snapshot < snapshot_count; ++snapshot)
        {
            copy_rows(
                *this->input_BC,
                boundary,
                snapshot * boundary_rows,
                static_cast<T>(time_step * snapshot));
        }

        VectorX<T> boundary_min(domain_dimension);
        VectorX<T> boundary_max(domain_dimension);
        boundary_min << T(0), T(0), T(0);
        boundary_max << T(2), T(2), T(2);
        this->input_BC->set_domain_params(boundary_min, boundary_max);

        const mfa::BurgersBoundaryAudit<T> audit =
            mfa::auditBurgersBoundaryPointSet(*this->input_BC);
        if (std::getenv("MFA_BURGERS_BC_DESIGN_AUDIT") != NULL)
        {
            std::cerr << "burgers_bc_design_audit=PASS rows=" << audit.rows
                      << " bottom=" << audit.face_counts[0]
                      << " left=" << audit.face_counts[1]
                      << " right=" << audit.face_counts[2]
                      << " top=" << audit.face_counts[3]
                      << " unique=" << audit.unique_physical_space_time_points
                      << " max_param_error=" << std::setprecision(17)
                      << audit.max_parameter_error << std::endl;
        }

        this->setup_MFA(proxy, mfa_info);
    }

private:
    static std::string join_path(
        const std::string& directory,
        const std::string& filename)
    {
        if (directory.empty() || directory == ".")
            return filename;
        if (directory[directory.size() - 1] == '/')
            return directory + filename;
        return directory + "/" + filename;
    }

    static std::string snapshot_path(
        const std::string& directory,
        int snapshot)
    {
        std::ostringstream filename;
        filename << "uv_times_" << std::setw(4) << std::setfill('0')
                 << snapshot << ".txt";
        return join_path(directory, filename.str());
    }

    static std::vector<SpatialRow> read_rows(
        const std::string& path,
        int expected_rows)
    {
        std::ifstream input_file(path.c_str());
        if (!input_file)
            throw std::runtime_error("Cannot open Burgers input file: " + path);

        std::vector<SpatialRow> rows(static_cast<std::size_t>(expected_rows));
        for (int row = 0; row < expected_rows; ++row)
        {
            SpatialRow& values = rows[static_cast<std::size_t>(row)];
            if (!(input_file >> values[0] >> values[1] >> values[2] >> values[3]) ||
                !std::isfinite(static_cast<double>(values[0])) ||
                !std::isfinite(static_cast<double>(values[1])) ||
                !std::isfinite(static_cast<double>(values[2])) ||
                !std::isfinite(static_cast<double>(values[3])))
            {
                throw std::runtime_error(
                    "Malformed or truncated Burgers input file at row " +
                    std::to_string(row + 1) + ": " + path);
            }
        }

        std::string trailing_token;
        if (input_file >> trailing_token)
            throw std::runtime_error("Unexpected extra data in Burgers input file: " + path);
        return rows;
    }

    static void copy_rows(
        mfa::PointSet<T>& destination,
        const std::vector<SpatialRow>& rows,
        int destination_offset,
        T time)
    {
        for (std::size_t row = 0; row < rows.size(); ++row)
        {
            const int output_row = destination_offset + static_cast<int>(row);
            destination.domain(output_row, 0) = static_cast<T>(rows[row][0]);
            destination.domain(output_row, 1) = static_cast<T>(rows[row][1]);
            destination.domain(output_row, 2) = time;
            destination.domain(output_row, 3) = static_cast<T>(rows[row][2]);
            destination.domain(output_row, 4) = static_cast<T>(rows[row][3]);
        }
    }

    void set_physical_bounds(T minimum, T maximum)
    {
        for (int dimension = 0; dimension < 3; ++dimension)
        {
            this->bounds_mins(dimension) = minimum;
            this->bounds_maxs(dimension) = maximum;
        }
    }
};

#endif
