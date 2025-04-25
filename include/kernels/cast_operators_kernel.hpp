#pragma once
#include "kernels/constants.hpp"
#include "utils/macros.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {

    template<typename T_src, typename T_dist>
    void launch_numerical_cast(T_src* src, T_dist* dist, const unsigned int sz) ;

    template<typename T_src, typename T_dist>
    __global__ void numerical_cast_kernel(T_src* src, T_dist* dist, const size_t sz) ;

}