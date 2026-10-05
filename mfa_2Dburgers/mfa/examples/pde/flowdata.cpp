#include <mfa/mfa.hpp>

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "block.hpp"
#include "opts.h"

namespace
{
constexpr int kDomainDimension = 3;
constexpr int kPointDimension = 5;

void validate_control_points(
    int degree,
    std::vector<int>& control_points)
{
    if (control_points.size() == 1)
        control_points.assign(kDomainDimension, control_points.front());
    if (control_points.size() != kDomainDimension)
        throw std::invalid_argument(
            "Burger2D requires one or three -v/--vars_nctrl values");
    for (std::size_t dimension = 0;
         dimension < control_points.size();
         ++dimension)
    {
        if (control_points[dimension] <= degree)
            throw std::invalid_argument(
                "Each variable control-point count must exceed the degree");
    }
}

mfa::MFAInfo make_mfa_info(
    int geometry_degree,
    int geometry_control_points,
    int variable_degree,
    const std::vector<int>& variable_control_points,
    real_t regularization,
    int regularize_first_and_second,
    int verbose)
{
    if (geometry_control_points < 0)
        geometry_control_points = geometry_degree + 1;
    if (geometry_control_points <= geometry_degree)
        throw std::invalid_argument(
            "The geometry control-point count must exceed the degree");

    mfa::MFAInfo info(kDomainDimension, verbose);
    info.addGeomInfo(mfa::ModelInfo(
        kDomainDimension,
        kDomainDimension,
        geometry_degree,
        geometry_control_points));
    info.addVarInfo(mfa::ModelInfo(
        kDomainDimension,
        1,
        variable_degree,
        variable_control_points));
    info.addVarInfo(mfa::ModelInfo(
        kDomainDimension,
        1,
        variable_degree,
        variable_control_points));
    info.verbose = verbose;
    info.weighted = 0;
    info.regularization = regularization;
    info.reg1and2 = regularize_first_and_second;
    return info;
}
}

int main(int argc, char** argv)
{
    diy::mpi::environment environment(argc, argv);
    diy::mpi::communicator world;

    int geometry_degree = 2;
    int variable_degree = 2;
    int geometry_control_points = -1;
    std::vector<int> variable_control_points(1, 11);
    std::string input = "Burger2D";
    std::string input_directory = ".";
    real_t regularization = 0;
    int regularize_first_and_second = 0;
    int verbose = 1;
    bool help = false;

    int structured = 1;

    opts::Options options;
    options >> opts::Option('i', "input", input, " input dataset (Burger2D)");
    options >> opts::Option('f', "infile", input_directory, " input data directory");
    options >> opts::Option('p', "geom_degree", geometry_degree, " geometry degree");
    options >> opts::Option('q', "vars_degree", variable_degree, " solution degree");
    options >> opts::Option(
        'g', "geom_nctrl", geometry_control_points,
        " geometry control points per dimension");
    options >> opts::Option(
        'v', "vars_nctrl", variable_control_points,
        " solution control points; repeat once per dimension");
    options >> opts::Option(
        'b', "regularization", regularization,
        " legacy regularization strength (zero in the selected run)");
    options >> opts::Option(
        'k', "reg1and2", regularize_first_and_second,
        " legacy derivative-order switch");
    options >> opts::Option('V', "verbose", verbose, " output verbosity");
    options >> opts::Option(
        'x', "structured", structured,
        " structured-input flag; must be one");
    options >> opts::Option('h', "help", help, " show help");

    if (!options.parse(argc, argv) || help)
    {
        if (world.rank() == 0)
            std::cout << options;
        return help ? 0 : 2;
    }

    try
    {
        if (input != "Burger2D")
            throw std::invalid_argument(
                "This executable supports only the Burger2D dataset");
        if (structured != 1)
            throw std::invalid_argument(
                "Burger2D requires structured input (-x 1)");
        if (geometry_degree < 0 || variable_degree < 0)
            throw std::invalid_argument("Spline degrees must be nonnegative");

        validate_control_points(variable_degree, variable_control_points);
        mfa::MFAInfo mfa_info = make_mfa_info(
            geometry_degree,
            geometry_control_points,
            variable_degree,
            variable_control_points,
            regularization,
            regularize_first_and_second,
            verbose);

        if (world.rank() == 0)
        {
            std::cerr << "PI-BSR two-dimensional Burgers fit\n"
                      << "  data directory: " << input_directory << '\n'
                      << "  degree: " << variable_degree << '\n'
                      << "  control points: "
                      << mfa::print_vec(variable_control_points) << '\n';
        }

        const int total_blocks = 1;
        const int memory_blocks = -1;
        const int threads = 1;
        diy::FileStorage storage("./DIY.XXXXXX");
        diy::Master master(
            world,
            threads,
            memory_blocks,
            &Block<real_t>::create,
            &Block<real_t>::destroy,
            &storage,
            &Block<real_t>::save,
            &Block<real_t>::load);
        diy::ContiguousAssigner assigner(world.size(), total_blocks);

        Bounds<real_t> domain_bounds(kDomainDimension);
        domain_bounds.min = {0, 0, 0};
        domain_bounds.max = {2, 2, 2};
        Decomposer<real_t> decomposer(
            kDomainDimension,
            domain_bounds,
            total_blocks);
        decomposer.decompose(
            world.rank(),
            assigner,
            [&](int gid,
                const Bounds<real_t>& core,
                const Bounds<real_t>& bounds,
                const Bounds<real_t>& domain,
                const RCLink<real_t>& link)
            {
                Block<real_t>::add(
                    gid,
                    core,
                    bounds,
                    domain,
                    link,
                    master,
                    kDomainDimension,
                    kPointDimension);
            });

        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->read_burgers_data(proxy, mfa_info, input_directory);
            });

        std::cerr << "\nStarting fixed encoding...\n\n";
        double encode_time = MPI_Wtime();
        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->fixed_encode_block(proxy, mfa_info);
            });
        encode_time = MPI_Wtime() - encode_time;

        double decode_time = MPI_Wtime();
        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->range_error(proxy, true, false);
            });
        decode_time = MPI_Wtime() - decode_time;

        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->print_block(proxy, true);
            });
        std::cerr << "encoding time = " << encode_time << " s\n"
                  << "decoding time = " << decode_time << " s\n";

        diy::io::write_blocks("approx.mfa", world, master);
    }
    catch (const std::exception& error)
    {
        if (world.rank() == 0)
            std::cerr << "Burger2D run failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
