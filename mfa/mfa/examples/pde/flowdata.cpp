#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>

#include "block.hpp"
#include "example-setup.hpp"
#include "opts.h"

int main(int argc, char** argv)
{
    diy::mpi::environment environment(argc, argv);
    diy::mpi::communicator world;

    int geom_degree = 3;
    int vars_degree = 2;
    int geom_nctrl = -1;
    std::vector<int> vars_nctrl(1, 11);
    std::string input = "ConDiff1D";
    std::string data_directory = ".";
    real_t regularization = 0;
    int reg1and2 = 0;
    int structured = 1;
    int verbose = 1;
    bool help = false;

    opts::Options options;
    options >> opts::Option(
        'p', "geom_degree", geom_degree, "geometry spline degree");
    options >> opts::Option(
        'q', "vars_degree", vars_degree, "science-variable spline degree");
    options >> opts::Option(
        'g', "geom_nctrl", geom_nctrl, "geometry control count per axis");
    options >> opts::Option(
        'v', "vars_nctrl", vars_nctrl,
        "science-variable control count; provide once per axis");
    options >> opts::Option('i', "input", input, "dataset identifier");
    options >> opts::Option(
        'f', "infile", data_directory, "directory containing packaged data");
    options >> opts::Option(
        'b', "regularization", regularization, "MFA smoothing parameter");
    options >> opts::Option(
        'k', "reg1and2", reg1and2,
        "regularize first and second derivatives when set to 1");
    options >> opts::Option(
        'x', "structured", structured, "structured-input flag (must be 1)");
    options >> opts::Option('V', "verbose", verbose, "output verbosity");
    options >> opts::Option('h', "help", help, "show this help message");

    if (!options.parse(argc, argv))
    {
        if (world.rank() == 0)
            std::cerr << options;
        return 2;
    }
    if (help)
    {
        if (world.rank() == 0)
            std::cout << options;
        return 0;
    }

    try
    {
        if (input != "ConDiff1D")
            throw std::invalid_argument(
                "This reviewer target supports only -i ConDiff1D");
        if (structured != 1)
            throw std::invalid_argument(
                "ConDiff1D requires structured input (-x 1)");

        print_convection_diffusion_settings(
            geom_degree,
            geom_nctrl,
            vars_degree,
            vars_nctrl,
            data_directory,
            regularization,
            reg1and2);

        const int dom_dim = 2;
        const int pt_dim = 3;
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

        Bounds<real_t> domain_bounds(dom_dim);
        set_convection_diffusion_bounds(domain_bounds);
        Decomposer<real_t> decomposer(dom_dim, domain_bounds, total_blocks);
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
                    dom_dim,
                    pt_dim);
            });

        const std::vector<int> model_dims = {2, 1};
        mfa::MFAInfo mfa_info(dom_dim, verbose);
        DomainArgs data_args(dom_dim, model_dims);
        setup_convection_diffusion(
            geom_degree,
            geom_nctrl,
            vars_degree,
            vars_nctrl,
            data_directory,
            regularization,
            reg1and2,
            verbose,
            mfa_info,
            data_args);

        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->read_convection_diffusion_data(
                    proxy, mfa_info, data_args);
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
        std::cerr << "\nEncoding done.\n"
                  << "\nFinal decoding and computing maximum error...\n";

        double decode_time = MPI_Wtime();
        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->range_error(proxy, true, false);
            });
        decode_time = MPI_Wtime() - decode_time;

        std::cerr << "\n------- Final block results --------\n";
        master.foreach(
            [&](Block<real_t>* block,
                const diy::Master::ProxyWithLink& proxy)
            {
                block->print_block(proxy, true);
            });
        std::cerr << "encoding time         = " << encode_time << " s.\n"
                  << "decoding time         = " << decode_time << " s.\n"
                  << "-------------------------------------\n";

        diy::io::write_blocks("approx.mfa", world, master);
    }
    catch (const std::exception& error)
    {
        if (world.rank() == 0)
            std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
