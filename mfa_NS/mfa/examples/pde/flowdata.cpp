#include <mfa/mfa.hpp>

#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/block.hpp>
#include <diy/master.hpp>

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "block.hpp"
#include "example-setup.hpp"
#include "opts.h"

int main(int argc, char** argv)
{
    diy::mpi::environment environment(argc, argv);
    diy::mpi::communicator world;

    std::string input = "NS2D";
    std::string infile = ".";
    int vars_degree = 3;
    std::vector<int> vars_nctrl = {11};
    real_t regularization = 0;
    int reg1and2 = 0;
    int structured = 1;
    int verbose = 1;
    bool help = false;

    opts::Options options;
    options >> opts::Option('i', "input", input, "input dataset; must be NS2D");
    options >> opts::Option('f', "infile", infile, "cavity data directory");
    options >> opts::Option('q', "vars_degree", vars_degree, "science-variable degree");
    options >> opts::Option('v', "vars_nctrl", vars_nctrl, "science-variable control points per dimension");
    options >> opts::Option('b', "regularization", regularization, "legacy regularization value; selected run uses zero");
    options >> opts::Option('k', "reg1and2", reg1and2, "legacy derivative-order switch; selected run uses zero");
    options >> opts::Option('x', "structured", structured, "structured-input flag; must be one");
    options >> opts::Option('V', "verbose", verbose, "output verbosity");
    options >> opts::Option('h', "help", help, "show help");

    if (!options.parse(argc, argv))
    {
        if (world.rank() == 0)
            std::cout << options;
        return 1;
    }
    if (help)
    {
        if (world.rank() == 0)
            std::cout << options;
        return 0;
    }
    if (input != "NS2D" || structured != 1)
    {
        if (world.rank() == 0)
            std::cerr << "flowdata accepts only structured NS2D input.\n";
        return 1;
    }

    try
    {
        echo_mfa_settings(
            vars_degree,
            vars_nctrl,
            regularization,
            reg1and2,
            verbose);
        echo_data_settings(infile);

        const int dom_dim = 3;
        const int pt_dim = 6;
        const int total_blocks = 1;
        const int memory_blocks = -1;
        const int thread_count = 1;

        Bounds<real_t> domain_bounds(dom_dim);
        set_ns2d_bounds(domain_bounds);

        diy::FileStorage storage("./DIY.XXXXXX");
        diy::Master master(
            world,
            thread_count,
            memory_blocks,
            &Block<real_t>::create,
            &Block<real_t>::destroy,
            &storage,
            &Block<real_t>::save,
            &Block<real_t>::load);
        diy::ContiguousAssigner assigner(world.size(), total_blocks);
        Decomposer<real_t> decomposer(dom_dim, domain_bounds, total_blocks);
        decomposer.decompose(
            world.rank(),
            assigner,
            [&](
                int gid,
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

        mfa::MFAInfo mfa_info(dom_dim, verbose);
        DomainArgs domain_args;
        setup_ns2d(
            vars_degree,
            vars_nctrl,
            infile,
            reg1and2,
            regularization,
            verbose,
            mfa_info,
            domain_args);

        master.foreach(
            [&](Block<real_t>* block, const diy::Master::ProxyWithLink& proxy)
            {
                block->read_2d_structured_NSdata(proxy, mfa_info, domain_args);
            });

        std::fprintf(stderr, "\nStarting fixed encoding...\n\n");
        double encode_time = MPI_Wtime();
        master.foreach(
            [&](Block<real_t>* block, const diy::Master::ProxyWithLink& proxy)
            {
                block->fixed_encode_block(proxy, mfa_info);
            });
        encode_time = MPI_Wtime() - encode_time;
        std::fprintf(stderr, "\nEncoding done.\n");

        std::fprintf(stderr, "\nFinal decoding and computing max. error...\n");
        double decode_time = MPI_Wtime();
        master.foreach(
            [&](Block<real_t>* block, const diy::Master::ProxyWithLink& proxy)
            {
                block->range_error(proxy, true, false);
            });
        decode_time = MPI_Wtime() - decode_time;

        std::fprintf(stderr, "\n------- Final block results --------\n");
        master.foreach(
            [&](Block<real_t>* block, const diy::Master::ProxyWithLink& proxy)
            {
                block->print_block(proxy, true);
            });
        std::fprintf(stderr, "encoding time         = %.3lf s.\n", encode_time);
        std::fprintf(stderr, "decoding time         = %.3lf s.\n", decode_time);
        std::fprintf(stderr, "-------------------------------------\n");

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
