#pragma once

#include "utils/macros.hpp"
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {


    enum class OperandType  : __uint8_t {
        SCALAR = 0,
        VECTOR,
    };

    template <typename T, typename Op>
    void launch_binary_operators(T* d_rhs, T* d_lhs, OperandType t_rhs, OperandType t_lhs, T* d_res, unsigned int sz) ;

    template <typename T, typename Op>
    __global__ void apply_batches(T* rhs, T* lhs, T* res, const unsigned int sz);

    // Functor base class for binary operators
    template <typename T>
    struct BinaryOperator {
        __device__ virtual T apply(T a, T b) const = 0;
    };
    
    template <typename T>
    struct AddOperator : BinaryOperator<T> {
        __device__ T apply(T a, T b) const override { return a + b; }
    };
    
    template <typename T>
    struct MinusOperator : BinaryOperator<T> {
        __device__ T apply(T a, T b) const override { return a - b; }
    };
    

    template <typename T>
    struct MulOperator : BinaryOperator<T> {
        __device__ T apply(T a, T b) const override { return a * b; }
    };
    
    template <typename T>
    struct DivOperator : BinaryOperator<T> {
        __device__ T apply(T a, T b) const override { return b != 0 ? a / b : 0; } 
    };
    //TODO: it should only apply to int
    template <typename T>
    struct RemOperator : BinaryOperator<T> {
        __device__ T apply(T a, T b) const override { return b != 0 ? a % b : 0; } 
    };
    



}