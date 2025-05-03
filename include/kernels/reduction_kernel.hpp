#pragma once
#include "utils/macros.hpp"
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {
    void launch_convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output);
    
    void launch_sum_double_precision(float* __restrict__ d_arr, double* __restrict__ res,  const uint32_t sz, cudaStream_t& stream, const double inital);

    template <typename T, typename Op>
    void launch_reduction_operators(T* __restrict__ d_arr, T* __restrict__ res, const uint32_t sz, cudaStream_t& stream, const T inital) ;


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