#include "kernels/reduction_kernel.hpp"

namespace YallaSQL::Kernel
{
    __global__ void convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *output = static_cast<float>(*input);
        }
    }

    void launch_convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output) {
        convert_double_to_float_kernel<<<1, 1>>>(input, output);
    }


    __global__ void sum_double_precision(float* __restrict__ arr, double* __restrict__ res, char* __restrict__ isnull, const uint32_t sz, const double inital) {
        __shared__ double wrap_reductions[BLOCK_DIM/32];
        __shared__ double sh_res;
        if(threadIdx.x == 0)  
            sh_res = inital;
        // get info
        int tid = threadIdx.x;
        int wrapIdx = tid/32;
        int laneIdx = tid%32;

        double localVal = inital;
        int globalIdx = threadIdx.x + COARSENING_FACTOR * blockIdx.x * blockDim.x;
        #pragma unroll
        for(uint32_t k = 0;k < COARSENING_FACTOR;k++) {
            // 1. get local value for each ele in wrap
            localVal = globalIdx < sz && !isnull[globalIdx] ? arr[globalIdx] : inital;
            // 2. each wrap reduce
            #pragma unroll
            for(uint32_t d = 16; d > 0; d>>=1){
                double neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                localVal = localVal + neighbor;
            }
            // each wrap put it's value
            if(laneIdx == 0)  
                wrap_reductions[wrapIdx] = localVal;
            __syncthreads();
            // reduce out of each wrap
            if(wrapIdx == 0) {
                localVal = tid < BLOCK_DIM/32 ? wrap_reductions[tid] : inital;
                #pragma unroll
                for(uint32_t d = 16; d > 0; d>>=1) {
                    double neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                    localVal = localVal +  neighbor;
                }
                if(laneIdx == 0) 
                    sh_res = localVal + sh_res;
            }
            globalIdx += blockDim.x;
        }
        if(threadIdx.x == 0)
            atomicAdd(res, sh_res);
    }

    void launch_sum_double_precision(float* __restrict__ d_arr, double* __restrict__ res, char* __restrict__ isnull,  const uint32_t sz, cudaStream_t& stream, const double inital) {
        dim3 threads(BLOCK_DIM);
        dim3 blocks(CEIL_DIV(sz, COARSENING_FACTOR*BLOCK_DIM));

        sum_double_precision<<<blocks, threads, 0, stream>>>(d_arr, res, isnull, sz, inital);
        CUDA_CHECK_LAST();

    }

}