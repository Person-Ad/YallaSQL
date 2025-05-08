#pragma once
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {

    #define ELE_PER_TH 6
    #define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

    template <typename T>
    __device__ int find_corank(T* A, T* B, uint32_t m, uint32_t n, uint32_t k);

    // template <typename T>
    // __global__ void merge_sorted_array_kernel(T* A, T* B, T* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n);
    
    template <typename T>
    void launch_merge_sorted_array_kernel(T* A, T* B, T* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n, cudaStream_t stream);

    void launch_merge_sorted_array_kernel_str(String* A, String* B, String* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n, cudaStream_t stream) ;
}