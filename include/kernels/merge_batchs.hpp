#pragma once
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {

    #define ELE_PER_TH 6
    #define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

    __device__ int find_corank(int* A, int* B, uint32_t m, uint32_t n, uint32_t k);

    __global__ void merge_sorted_array_kernel(int* A, int* B, int* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n);

    void launch_merge_sorted_array_kernel(int* A, int* B, int* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n, cudaStream_t stream);
}