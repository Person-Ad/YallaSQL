#include "kernels/cross_product_kernel.hpp"

namespace YallaSQL::Kernel
{
    #define COARSENING_FACTOR_CROSS 5

    template <typename T>
    __global__ void cross_product_kernel(const T* __restrict__ col, T* __restrict__ out_col, const char*  __restrict__ src_nullset, char*  __restrict__ out_nullset, const uint32_t leftBs, const uint32_t rightBs, const bool isleft) {
        uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t global_idy = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t stride = blockDim.x * gridDim.x;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR_CROSS; k++) {
            if(global_idx < leftBs && global_idy < rightBs && ((isleft && global_idx < leftBs) || (!isleft && global_idy < rightBs))) {
                int output_row = global_idx * rightBs + global_idy;
                if(isleft) {
                    out_col[output_row] = col[global_idx];
                    out_nullset[output_row] = src_nullset[global_idx];
                } else {
                    out_col[output_row] = col[global_idy];
                    out_nullset[output_row] = src_nullset[global_idy];
                }
            }
            global_idx += stride;
        }
    }

    template <typename T>
    void launch_cross_product_col(const T* __restrict__ col, T* __restrict__ out_col, const char*  __restrict__ src_nullset, char*  __restrict__ out_nullset, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft) {
        dim3 threads(32, 32);
        dim3 blocks(CEIL_DIV(leftBs, COARSENING_FACTOR_CROSS * threads.x), CEIL_DIV(rightBs, threads.y));
        cross_product_kernel<<<blocks, threads, 0, stream>>>(col, out_col, src_nullset, out_nullset, leftBs, rightBs, isleft);
    }

    
    template void launch_cross_product_col<int>    (const int* __restrict__, int* __restrict__ out_col_right, const char* __restrict__, char* __restrict__, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft); 
    template void launch_cross_product_col<float>  (const float* __restrict__, float* __restrict__ out_col_right, const char* __restrict__, char* __restrict__, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft); 
    template void launch_cross_product_col<int64_t>(const int64_t* __restrict__, int64_t* __restrict__ out_col_right, const char* __restrict__, char* __restrict__, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft); 
    template void launch_cross_product_col<String> (const String* __restrict__ right_col, String* __restrict__ out_col_right, const char* __restrict__, char* __restrict__, const uint32_t leftBs, const uint32_t rightBs, cudaStream_t stream, const bool isleft); 

}