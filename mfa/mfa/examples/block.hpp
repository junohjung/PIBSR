#ifndef PIBSR_CONDIFF_BLOCK_HPP
#define PIBSR_CONDIFF_BLOCK_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mfa/block_base.hpp>
#include <mfa/mfa.hpp>

#include "domain_args.hpp"

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
    using Base::input_BC;
    using Base::input_IC;
    using Base::pt_dim;

    Block()
        : Base()
    {
        this->input_orig = nullptr;
        input_BC = nullptr;
        input_IC = nullptr;
    }

    ~Block()
    {
        delete input_BC;
        delete input_IC;
    }

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
        int block_dom_dim,
        int block_pt_dim,
        T ghost_factor = T(0))
    {
        mfa::add<Block, T>(
            gid,
            core,
            bounds,
            domain,
            link,
            master,
            block_dom_dim,
            block_pt_dim,
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

    void read_convection_diffusion_data(
        const diy::Master::ProxyWithLink& proxy,
        mfa::MFAInfo& mfa_info,
        const DomainArgs& args)
    {
        if (dom_dim != 2 || pt_dim != 3 ||
            mfa_info.dom_dim != 2 || mfa_info.pt_dim() != 3 ||
            mfa_info.geom_dim() != 2 || mfa_info.nvars() != 1 ||
            mfa_info.var_dim(0) != 1)
            throw std::invalid_argument(
                "The convection-diffusion reader requires model dimensions {2, 1}");

        const int space_points = 60;
        const int time_points = 400;
        const T final_time = T(0.4);
        const T time_step = final_time / T(time_points - 1);
        const VectorXi model_dims = mfa_info.model_dims();

        this->max_errs.resize(1);
        this->sum_sq_errs.resize(1);
        core_mins.resize(2);
        core_maxs.resize(2);
        bounds_mins.resize(3);
        bounds_maxs.resize(3);

        VectorXi trajectory_shape(2);
        trajectory_shape << space_points, time_points;
        input = new mfa::PointSet<T>(
            2,
            model_dims,
            space_points * time_points,
            trajectory_shape);

        for (int time_index = 0; time_index < time_points; ++time_index)
        {
            std::ostringstream name;
            name << "rho_T3_";
            name.width(4);
            name.fill('0');
            name << time_index << ".txt";
            const std::vector<T> values = read_values(
                join_path(args.infile, name.str()),
                static_cast<std::size_t>(space_points * 2));

            for (int space_index = 0; space_index < space_points; ++space_index)
            {
                const int row = time_index * space_points + space_index;
                input->domain(row, 0) = values[2 * space_index];
                input->domain(row, 1) = time_step * T(time_index);
                input->domain(row, 2) = values[2 * space_index + 1];
            }
        }

        bounds_mins.head(2) << args.min[0], args.min[1];
        bounds_maxs.head(2) << args.max[0], args.max[1];
        bounds_mins(2) = input->domain.col(2).minCoeff();
        bounds_maxs(2) = input->domain.col(2).maxCoeff();
        core_mins = bounds_mins.head(2);
        core_maxs = bounds_maxs.head(2);
        input->set_domain_params();

        VectorXi initial_shape(2);
        initial_shape << space_points, 1;
        input_IC = new mfa::PointSet<T>(
            2,
            model_dims,
            space_points,
            initial_shape);
        const std::vector<T> initial_values = read_values(
            join_path(args.infile, "IC.txt"),
            static_cast<std::size_t>(space_points * 2));
        for (int space_index = 0; space_index < space_points; ++space_index)
        {
            input_IC->domain(space_index, 0) = initial_values[2 * space_index];
            input_IC->domain(space_index, 1) = T(0);
            input_IC->domain(space_index, 2) = initial_values[2 * space_index + 1];
        }
        input_IC->set_domain_params();

        VectorXi boundary_shape(2);
        boundary_shape << 2, time_points;
        input_BC = new mfa::PointSet<T>(
            2,
            model_dims,
            2 * time_points,
            boundary_shape);
        const std::vector<T> boundary_values = read_values(
            join_path(args.infile, "BC.txt"), 4);
        for (int time_index = 0; time_index < time_points; ++time_index)
        {
            for (int boundary_index = 0; boundary_index < 2; ++boundary_index)
            {
                const int row = 2 * time_index + boundary_index;
                input_BC->domain(row, 0) = boundary_values[2 * boundary_index];
                input_BC->domain(row, 1) = time_step * T(time_index);
                input_BC->domain(row, 2) = boundary_values[2 * boundary_index + 1];
            }
        }
        input_BC->set_domain_params();

        this->setup_MFA(proxy, mfa_info);
    }

  private:
    static std::string join_path(
        const std::string& directory,
        const std::string& filename)
    {
        if (directory.empty() || directory == ".")
            return filename;
        const char last = directory[directory.size() - 1];
        if (last == '/' || last == '\\')
            return directory + filename;
        return directory + "/" + filename;
    }

    static std::vector<T> read_values(
        const std::string& path,
        std::size_t expected_count)
    {
        std::ifstream stream(path.c_str());
        if (!stream)
            throw std::runtime_error("Could not open input file: " + path);

        std::vector<T> values(expected_count);
        for (std::size_t index = 0; index < expected_count; ++index)
        {
            if (!(stream >> values[index]) || !std::isfinite(values[index]))
            {
                std::ostringstream message;
                message << "Expected " << expected_count
                        << " finite numeric values in " << path
                        << "; failed at value " << (index + 1);
                throw std::runtime_error(message.str());
            }
        }

        stream >> std::ws;
        if (stream.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error(
                "Unexpected trailing content in input file: " + path);

        return values;
    }
};

#endif
