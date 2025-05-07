#pragma once

#include <cuda_runtime.h>
#include <stdint-gcc.h>
#include "kernels/constants.hpp"
#include "kernels/string_kernel.hpp"
#include "utils/macros.hpp"

#define OUR_RADIX 4
#define BUCKET_SZ 16 // 1<<RADIX //! need to edit 
// get ith bit in x
#define GET_BIT(x,i) ((x >> i) & 1LL)
// get mask 000111 if radix = 3
#define MASK_ZERO ((1LL << (OUR_RADIX)) - 1LL)
// get mask of iter 000111000 if iter = 1
#define MASK_ITER(iter) (MASK_ZERO << (iter*OUR_RADIX))
// get radix for certain iter
#define GET_RADIX_KEY(x,iter) ((x>>(iter*OUR_RADIX)) & MASK_ZERO)

namespace YallaSQL::Kernel {

    template <typename OP, typename T, size_t BlockDim>
    void launch_radix_sort(T* d_input, uint32_t* &d_idxs_in, const uint32_t N, cudaStream_t stream);

    template <typename T>
    struct SortOp
    {
        __device__ virtual inline int get_radix(T& x, int iter) const = 0;
        __device__ virtual inline int get_bit(T& x, int bit) const = 0;
    };
    template <typename T>
    struct AscOp : SortOp<T>
    {
        __device__ inline int get_radix(T& x, int iter) const override { return GET_RADIX_KEY(x, iter);  }
        __device__ inline int get_bit(T& x, int bit) const override { return GET_BIT(x, bit); };
    };
    template <typename T>
    struct DescOp : SortOp<T>
    {
        __device__ inline int get_radix(T& x, int iter) const override { return MASK_ZERO ^ GET_RADIX_KEY(x, iter);  }

        __device__ inline int get_bit(T& x, int bit) const override { return (1LL ^ GET_BIT(x, bit)); };
    };

    struct AscOpString : SortOp<String>
    {
        //TODO: change it if RADIX!=8
        __device__ inline int get_radix(String& x, int iter) const override { 
            int bit_start = iter * OUR_RADIX;
            int byte_idx = MAX_STR - 1 - (bit_start / 8); // Right-to-left byte index
            int bit_offset = bit_start % 8; // Starting bit within the byte

            uint32_t radix_value = uint32_t(x.data[byte_idx]) >> bit_offset;
            radix_value &= (1 << OUR_RADIX) - 1; // Mask to RADIX_BITS
            return radix_value;
        }
        __device__ inline int get_bit(String& x, int bit) const override { return GET_BIT(x.data[MAX_STR - 1 - bit/8], bit%8); };
    };
    
    struct DescOpString : SortOp<String>
    {
        //TODO: change it if RADIX!=8
        __device__ inline int get_radix(String& x, int iter) const override {
            int bit_start = iter * OUR_RADIX;
            int byte_idx = MAX_STR - 1 - (bit_start / 8); // Right-to-left byte index
            int bit_offset = bit_start % 8; // Starting bit within the byte

            uint32_t radix_value = uint32_t(x.data[byte_idx]) >> bit_offset;
            radix_value &= (1 << OUR_RADIX) - 1; // Mask to RADIX_BITS
            return MASK_ZERO ^ radix_value;
        }
        __device__ inline int get_bit(String& x, int bit) const override { return 1LL ^ GET_BIT(x.data[MAX_STR - 1 - bit/8], bit%8);};
    };

    __global__  void int_to_uint32(int* arr, uint32_t* res, int N) ;

    __global__  void uint32_to_int(uint32_t* arr, int* res, int N) ;

    __global__  void float_to_uint32(float* arr, uint32_t* res, int N) ;

    __global__  void uint32_to_float(uint32_t* arr, float* res, int N) ;

    void launch_int_to_uint32(int* arr, uint32_t* res, int N) ;

    void launch_uint32_to_int(uint32_t* arr, int* res, int N) ;

    void launch_float_to_uint32(float* arr, uint32_t* res, int N) ;

    void launch_uint32_to_float(uint32_t* arr, float* res, int N) ;

}