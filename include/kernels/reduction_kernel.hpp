#pragma once
#include "utils/macros.hpp"
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {
    void launch_reduction_count_notnull(char* __restrict__ d_arr, int* __restrict__ res, char* __restrict__ isnull,  const uint32_t sz, cudaStream_t& stream, const int inital);
    
    void launch_convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output);
    
    template <typename T>
    void launch_div_avg(const int* __restrict__ counter, const T* __restrict__ sum, float* __restrict__ avg);

    template <typename T, typename T_res, typename Op>
    void launch_reduction_operators(T* __restrict__ d_arr, T_res* __restrict__ res, char* __restrict__ isnull, const uint32_t sz, cudaStream_t& stream, const T_res inital) ;


    template <typename T>
    struct AggOperator {
        __device__ virtual T apply(T &a, T &b) const = 0;
    };

    template <typename T>
    struct MaxOperator : AggOperator<T> {
        __device__ T apply(T &a, T &b) const override { return a >= b ? a : b; }
    };

    template <typename T>
    struct MinOperator : AggOperator<T> {
        __device__ T apply(T &a, T &b) const override { return a <= b ? a : b; }
    };


    template <typename T>
    struct SumOperator : AggOperator<T> {
        __device__ T apply(T &a, T &b) const override { return a + b; }
    };

}