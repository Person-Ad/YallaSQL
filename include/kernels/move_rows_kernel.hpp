#pragma once

#include "utils/macros.hpp"
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {

    template<typename T>
    __global__ void move_rows_filter_kernel(T* __restrict__ src, T* res,
                                        uint32_t* __restrict__ map, // map[oldIdx] = newIdx + 1
                                        bool*    __restrict__ mask,
                                        const uint32_t srcSz) ;

    template<typename T>
    void launch_move_rows_filter_kernel(T* __restrict__ src, T* res, 
                                        uint32_t* __restrict__ map, 
                                        bool* __restrict__ mask, 
                                        const uint32_t srcSz, 
                                        cudaStream_t& stream);                                        


}