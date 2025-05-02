#pragma once

#include "kernels/binary_operators_kernel.hpp"
#include "kernels/string_kernel.hpp"

namespace YallaSQL::Kernel {
    template <typename T, typename Op>
    void launch_conditional_operators(T* __restrict__ d_rhs, T* __restrict__ d_lhs, OperandType t_rhs, OperandType t_lhs, bool*  d_res, unsigned int sz, cudaStream_t& stream, bool isneg = false);

    template <typename T>
    struct ConditionalOperator {
        __device__ virtual bool apply(T& a, T& b) const = 0;
    };

    template <typename T>
    struct LEOperator : ConditionalOperator<T> { // less than
        __device__ bool apply(T& a, T& b) const override { return a < b; }
    };

    template <typename T>
    struct LEQOperator : ConditionalOperator<T> { // less than or equal
        __device__ bool apply(T& a, T& b) const override { return a <= b; }
    };


    template <typename T>
    struct GEOperator : ConditionalOperator<T> { // greater than
        __device__ bool apply(T& a, T& b) const override { return a > b; }
    };

    template <typename T>
    struct GEQOperator : ConditionalOperator<T> { // greater than or equal
        __device__ bool apply(T& a, T& b) const override { return a >= b; }
    };


    template <typename T>
    struct EQOperator : ConditionalOperator<T> { // equal
        __device__ bool apply(T& a, T& b) const override { return a == b; }
    };

    template <typename T>
    struct NEQOperator : ConditionalOperator<T> { // equal
        __device__ bool apply(T& a, T& b) const override { return a != b; }
    };


    struct ANDOperator : ConditionalOperator<bool> { 
        __device__ bool apply(bool &a, bool &b) const override { return a && b; }
    };

    struct OROperator : ConditionalOperator<bool> { 
        __device__ bool apply(bool &a, bool &b) const override { return a || b; }
    };

    struct NOTOperator : ConditionalOperator<bool> { 
        __device__ bool apply(bool &a, bool &b) const override { return !a; }
    };


    template <>
    struct LEOperator<String> : ConditionalOperator<String> { // less than
        __device__ bool apply(String &a, String &b) const override { return strcmp_device(a, b) < 0; }
    };

    template <>
    struct LEQOperator<String> : ConditionalOperator<String> {
        __device__ bool apply(String &a, String &b) const override { return strcmp_device(a, b) <= 0;  }
    };

    template <>
    struct GEOperator<String> : ConditionalOperator<String> {
        __device__ bool apply(String &a, String &b) const override {  return strcmp_device(a, b) > 0; }
    };

    template <>
    struct GEQOperator<String> : ConditionalOperator<String> {
        __device__ bool apply(String &a, String &b) const override { return strcmp_device(a, b) >= 0; }
    };

    template <>
    struct EQOperator<String> : ConditionalOperator<String> {
        __device__ bool apply(String &a, String &b) const override { return strcmp_device(a, b) == 0; }
    };

    template <>
    struct NEQOperator<String> : ConditionalOperator<String> {
        __device__ bool apply(String &a, String &b) const override { return strcmp_device(a, b) != 0; }
    };

}