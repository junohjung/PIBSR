#ifndef PIBSR_CONDIFF_DOMAIN_ARGS_HPP
#define PIBSR_CONDIFF_DOMAIN_ARGS_HPP

#include <string>
#include <vector>

#include <mfa/types.hpp>

struct DomainArgs
{
    DomainArgs(int dom_dim, const std::vector<int>& dimensions)
        : model_dims(dimensions),
          min(static_cast<std::size_t>(dom_dim), real_t(0)),
          max(static_cast<std::size_t>(dom_dim), real_t(1))
    {
    }

    std::vector<int> model_dims;
    std::vector<real_t> min;
    std::vector<real_t> max;
    std::string infile;
};

#endif
