#ifndef PIBSR_CONDIFF_EXAMPLE_SETUP_HPP
#define PIBSR_CONDIFF_EXAMPLE_SETUP_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mfa/block_base.hpp>
#include <mfa/mfa.hpp>

#include "domain_args.hpp"

inline void set_convection_diffusion_bounds(Bounds<real_t>& bounds)
{
    if (bounds.min.dimension() != 2)
        throw std::invalid_argument(
            "The convection-diffusion example requires a two-dimensional x-t domain");

    bounds.min = {0.0, 0.0};
    bounds.max = {1.0, 0.4};
}

inline void setup_convection_diffusion(
    int geom_degree,
    int geom_nctrl,
    int vars_degree,
    std::vector<int> vars_nctrl,
    const std::string& data_directory,
    real_t regularization,
    int reg1and2,
    int verbose,
    mfa::MFAInfo& mfa_info,
    DomainArgs& args)
{
    if (vars_nctrl.size() == 1)
        vars_nctrl.assign(2, vars_nctrl.front());
    if (vars_nctrl.size() != 2)
        throw std::invalid_argument(
            "Provide exactly two science-variable control counts with -v");
    if (geom_nctrl < 0)
        geom_nctrl = geom_degree + 1;
    if (geom_nctrl <= geom_degree)
        throw std::invalid_argument(
            "The geometry control count must exceed the geometry degree");
    for (std::size_t axis = 0; axis < vars_nctrl.size(); ++axis)
    {
        if (vars_nctrl[axis] < 0)
            vars_nctrl[axis] = vars_degree + 1;
        if (vars_nctrl[axis] <= vars_degree)
            throw std::invalid_argument(
                "Each science-variable control count must exceed the degree");
    }

    args.infile = data_directory;
    args.min = {0.0, 0.0};
    args.max = {1.0, 0.4};

    mfa_info.reset();
    mfa_info.addGeomInfo(mfa::ModelInfo(2, 2, geom_degree, geom_nctrl));
    mfa_info.addVarInfo(mfa::ModelInfo(2, 1, vars_degree, vars_nctrl));
    mfa_info.verbose = verbose;
    mfa_info.weighted = 0;
    mfa_info.regularization = regularization;
    mfa_info.reg1and2 = reg1and2;
}

inline void print_convection_diffusion_settings(
    int geom_degree,
    int geom_nctrl,
    int vars_degree,
    const std::vector<int>& vars_nctrl,
    const std::string& data_directory,
    real_t regularization,
    int reg1and2,
    std::ostream& output = std::cerr)
{
    const int effective_geom_nctrl =
        geom_nctrl < 0 ? geom_degree + 1 : geom_nctrl;
    std::vector<int> effective_vars_nctrl = vars_nctrl;
    if (effective_vars_nctrl.size() == 1)
        effective_vars_nctrl.assign(2, effective_vars_nctrl.front());

    output << ">>> Running 'PI-BSR 1D convection-diffusion'\n\n"
           << "--------- MFA Settings ----------\n"
           << "pt_dim   = 3\tdom_dim    = 2\tscalar: true\n"
           << "geom_deg = " << geom_degree
           << "\tvars_deg = " << vars_degree << '\n'
           << "encoding type = fixed\n"
           << "geom_nctrl = " << effective_geom_nctrl
           << "\tvars_nctrl = " << mfa::print_vec(effective_vars_nctrl) << '\n'
           << "regularization = " << regularization << ", type: "
           << (regularization == 0
                   ? "N/A"
                   : (reg1and2 > 0 ? "1st and 2nd derivs" : "2nd derivs only"))
           << "\nweighted: false\n"
           << "parameterization method: domain\n"
           << "threading: serial\n"
           << "--------- Data Settings ----------\n"
           << "input: ConDiff1D, infile: " << data_directory << '\n';
}

#endif
