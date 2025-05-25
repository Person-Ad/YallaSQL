#pragma once
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {

    #define ELE_PER_TH 6
    #define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

    // Base struct for sort operations
    template <typename T>
    struct SortMergeOp {
        __device__ virtual bool is_less_or_equal(const T& a, const T& b) const = 0;
        __device__ virtual bool is_less_than(const T& a, const T& b) const = 0;
        __device__ virtual bool is_greater(const T& a, const T& b) const = 0;
        __device__ virtual bool is_greater_or_equal(const T& a, const T& b) const = 0;
    };

    // Ascending sort for numeric types
    template <typename T>
    struct AscSortMergeOp : SortMergeOp<T> {
        __device__ bool is_less_or_equal(const T& a, const T& b) const override { return a <= b; }
        __device__ bool is_less_than(const T& a, const T& b) const override { return a < b; }
        __device__ bool is_greater(const T& a, const T& b) const override { return a > b; }
        __device__ bool is_greater_or_equal(const T& a, const T& b) const override { return a >= b; }
    };

    template <typename T>
    struct DescSortMergeOp : SortMergeOp<T> {
        __device__ bool is_less_or_equal(const T& a, const T& b) const override { return a >= b; }
        __device__ bool is_less_than(const T& a, const T& b) const override { return a > b; }
        __device__ bool is_greater(const T& a, const T& b) const override { return a < b; }
        __device__ bool is_greater_or_equal(const T& a, const T& b) const override { return a <= b; }
    };
    
    struct AscSortStrOp : SortMergeOp<String> {
        __device__ bool is_less_or_equal(const String& a, const String& b) const override { return strcmp_device(a, b) <= 0; }
        __device__ bool is_less_than(const String& a, const String& b) const override { return strcmp_device(a, b) < 0; }
        __device__ bool is_greater(const String& a, const String& b) const override { return strcmp_device(a, b) > 0; }
        __device__ bool is_greater_or_equal(const String& a, const String& b) const override { return strcmp_device(a, b) >= 0; }
    };

    struct DescSortStrOp : SortMergeOp<String> {
        __device__ bool is_less_or_equal(const String& a, const String& b) const override { return strcmp_device(a, b) >= 0; }
        __device__ bool is_less_than(const String& a, const String& b) const override { return strcmp_device(a, b) > 0; }
        __device__ bool is_greater(const String& a, const String& b) const override { return strcmp_device(a, b) < 0; }
        __device__ bool is_greater_or_equal(const String& a, const String& b) const override { return strcmp_device(a, b) <= 0; }
    };

    template <typename T, typename OP>
    void launch_merge_sorted_array_kernel(T* A, T* B, T* C, uint32_t* new_idx, bool* table_idx, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n, cudaStream_t stream);
}