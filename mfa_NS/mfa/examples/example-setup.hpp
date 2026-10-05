#ifndef PIBSR_CAVITY_EXAMPLE_SETUP_HPP
#define PIBSR_CAVITY_EXAMPLE_SETUP_HPP

#include <mfa/mfa.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain_args.hpp"

inline void echo_mfa_settings(
    int vars_degree,
    const std::vector<int>& vars_nctrl,
    real_t regularization,
    int reg1and2,
    int verbose,
    std::ostream& output = std::cerr)
{
    if (verbose < 0)
        throw std::invalid_argument("verbosity must be nonnegative");

    output << ">>> Running 'lid-driven cavity'\n\n"
           << "--------- MFA Settings ----------\n"
           << "pt_dim   = 6\tdom_dim    = 3\tscalar: false\n"
           << "geom_deg = 2\tvars_deg = " << vars_degree << '\n'
           << "encoding type = fixed\n"
           << "geom_nctrl = 3\tvars_nctrl = "
           << mfa::print_vec(vars_nctrl) << '\n'
           << "regularization = " << regularization << ", type: "
           << (regularization == 0
                   ? "N/A"
                   : (reg1and2 > 0
                          ? "1st and 2nd derivs"
                          : "2nd derivs only"))
           << "\nweighted: false\n"
           << "parameterization method: domain\n"
           << "threading: serial\n";
}

inline void echo_data_settings(
    const std::string& infile,
    std::ostream& output = std::cerr)
{
    output << "--------- Data Settings ----------\n"
           << "input: NS2D, infile: " << infile << '\n'
           << "num pts    = 441441\ttest pts    = N/A\n";
}

inline void set_ns2d_bounds(Bounds<real_t>& bounds)
{
    if (bounds.min.dimension() != 3)
        throw std::invalid_argument("NS2D requires a three-dimensional domain");

    bounds.min = {0.0, 0.0, 0.0};
    bounds.max = {1.0, 1.0, 10.0};
}

inline void setup_ns2d(
    int vars_degree,
    std::vector<int> vars_nctrl,
    const std::string& infile,
    int reg1and2,
    real_t regularization,
    int verbose,
    mfa::MFAInfo& mfa_info,
    DomainArgs& args)
{
    const int dom_dim = 3;
    const std::vector<int> model_dims = {3, 1, 1, 1};

    if (vars_nctrl.size() == 1)
        vars_nctrl.assign(dom_dim, vars_nctrl.front());
    if (vars_nctrl.size() != static_cast<std::size_t>(dom_dim))
        throw std::invalid_argument(
            "NS2D requires one or three variable control-point counts");
    if (vars_degree < 0)
        throw std::invalid_argument("NS2D variable degree must be nonnegative");
    for (int control_points : vars_nctrl)
    {
        if (control_points <= vars_degree)
            throw std::invalid_argument(
                "Each NS2D control-point count must exceed the variable degree");
    }

    args.infile = infile;
    args.min = {0.0, 0.0, 0.0};
    args.max = {1.0, 1.0, 10.0};

    mfa_info.reset();
    mfa_info.addGeomInfo(mfa::ModelInfo(dom_dim, 3, 2, 3));
    for (std::size_t i = 1; i < model_dims.size(); ++i)
        mfa_info.addVarInfo(
            mfa::ModelInfo(dom_dim, model_dims[i], vars_degree, vars_nctrl));

    mfa_info.verbose = verbose;
    mfa_info.weighted = false;
    mfa_info.regularization = regularization;
    mfa_info.reg1and2 = reg1and2;
}

#endif
