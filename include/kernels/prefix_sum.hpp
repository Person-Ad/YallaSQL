#pragma once

#include "kernels/binary_operators_kernel.hpp"

namespace YallaSQL::Kernel {
    template <typename T>
    __device__ void _add_prev_block(T* res, T* sh_mem, 
        const int &actualBlockIdx, 
        const uint32_t &sz, 
        const uint32_t &arr_idx,
        uint32_t& blocks_finished);


    template <typename T>
    __device__ void upsweep_phase(T* sh_mem, uint64_t &offset);

    template <typename T>
    __device__ void downsweep_phase(T* sh_mem, uint64_t &offset);

    template <typename T>
    __global__ void prefix_sum(T* arr, T* res, 
                            const uint32_t sz,
                            uint32_t& blocks_counter,   // number should i take
                            uint32_t& blocks_finished); // number of blocks finished

    void launch_prefix_sum_mask(bool* arr, uint32_t* res, const uint32_t sz, cudaStream_t &stream);

    template <typename T>
    void launch_prefix_sum(T* arr, T* res, const uint32_t sz, cudaStream_t &stream);
}