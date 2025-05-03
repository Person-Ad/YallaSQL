#include "kernels/reduction_kernel.hpp"

namespace YallaSQL::Kernel
{

    __device__ void inline atomicReduction(int* addr, int& val, MaxOperator<int> &op) {
        atomicMax(addr, val);
    }
    __device__ void inline atomicReduction(int* addr, int& val, MinOperator<int> &op) {
        atomicMin(addr, val);
    }
    __device__ void inline atomicReduction(int* addr, int& val, SumOperator<int> &op) {
        atomicAdd(addr, val);
    }
    
    
    __device__ void inline atomicReduction(int64_t* addr, int64_t& val, SumOperator<int64_t> &op) {
        atomicAdd((unsigned long long*)addr, (unsigned long long)val);
    }
    __device__ void inline atomicReduction(int64_t* addr, int64_t& val, MaxOperator<int64_t> &op) {
        atomicMax((long long*)addr, (long long)val);
    }
    __device__ void inline atomicReduction(int64_t* addr, int64_t& val, MinOperator<int64_t> &op) {
        atomicMin((long long*)addr, (long long)val);
    }


    __device__ void inline atomicReduction(float* addr, float& val, SumOperator<float> &op) {
        atomicAdd(addr, val);
    }
    __device__ void inline atomicReduction(float* addr, float& val, MaxOperator<float> &op) {
        // automicMaxFloat(addr, val);
        int* address_as_i = (int*) addr;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }
    __device__ void inline atomicReduction(float* addr, float& val, MinOperator<float> &op) {
        // automicMaxFloat(addr, val);
        int* address_as_i = (int*) addr;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }


    template <typename T, typename Op>
    __global__ void reduction_sum_wrap(T* __restrict__ arr, T* __restrict__ res, const uint32_t sz, const T inital) {
        __shared__ T wrap_reductions[BLOCK_DIM/32];
        __shared__ T sh_res;
        Op op;
        if(threadIdx.x == 0)  
            sh_res = inital;
        // get info
        int tid = threadIdx.x;
        int wrapIdx = tid/32;
        int laneIdx = tid%32;

        T localVal = inital;
        int globalIdx = threadIdx.x + COARSENING_FACTOR * blockIdx.x * blockDim.x;
        #pragma unroll
        for(uint32_t k = 0;k < COARSENING_FACTOR;k++) {
            // 1. get local value for each ele in wrap
            localVal = globalIdx < sz ? arr[globalIdx] : inital;
            // 2. each wrap reduce
            #pragma unroll
            for(uint32_t d = 16; d > 0; d>>=1){
                T neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                localVal = op.apply(localVal, neighbor);
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
                    T neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                    localVal = op.apply(localVal,  neighbor );
                }
                if(laneIdx == 0) 
                    sh_res = op.apply(localVal, sh_res);
            }
            globalIdx += blockDim.x;
        }
        if(threadIdx.x == 0) atomicReduction(res, sh_res, op); 
            // op.atomicReduce(res, sh_res);
            // atomicAdd(res, sh_res);
    }

    template <typename T, typename Op>
    void launch_reduction_operators(T* __restrict__ d_arr, T* __restrict__ res,  const uint32_t sz, cudaStream_t& stream, const T inital) {
        dim3 threads(BLOCK_DIM);
        dim3 blocks(CEIL_DIV(sz, COARSENING_FACTOR*BLOCK_DIM));

        reduction_sum_wrap<T, Op><<<blocks, threads, 0, stream>>>(d_arr, res, sz, inital);
        CUDA_CHECK_LAST();

    }

    template void launch_reduction_operators<int, MaxOperator<int>>(int* __restrict__, int* __restrict__, const uint32_t, cudaStream_t&, const int);
    template void launch_reduction_operators<int, MinOperator<int>>(int* __restrict__, int* __restrict__, const uint32_t, cudaStream_t&, const int);
    template void launch_reduction_operators<int, SumOperator<int>>(int* __restrict__, int* __restrict__, const uint32_t, cudaStream_t&, const int);

    template void launch_reduction_operators<float, MaxOperator<float>>(float* __restrict__, float* __restrict__, const uint32_t, cudaStream_t&, const float);
    template void launch_reduction_operators<float, MinOperator<float>>(float* __restrict__, float* __restrict__, const uint32_t, cudaStream_t&, const float);
    template void launch_reduction_operators<float, SumOperator<float>>(float* __restrict__, float* __restrict__, const uint32_t, cudaStream_t&, const float);

    template void launch_reduction_operators<int64_t, MaxOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, const uint32_t, cudaStream_t&, const int64_t);
    template void launch_reduction_operators<int64_t, MinOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, const uint32_t, cudaStream_t&, const int64_t);
    template void launch_reduction_operators<int64_t, SumOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, const uint32_t, cudaStream_t&, const int64_t);

}