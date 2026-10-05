#ifndef PIBSR_CAVITY_DOMAIN_ARGS_HPP
#define PIBSR_CAVITY_DOMAIN_ARGS_HPP

#include <mfa/types.hpp>

#include <string>
#include <vector>

struct DomainArgs
{
    std::string infile;
    std::vector<real_t> min;
    std::vector<real_t> max;
};

#endif
