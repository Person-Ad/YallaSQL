#include "kernels/reduction_kernel.hpp"

namespace YallaSQL::Kernel
{
    template <typename T>
    __global__ void div_avg_kernel(const int* __restrict__ counter, const T* __restrict__ sum, float* __restrict__ avg) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *avg = static_cast<double>(*sum) / static_cast<double>(*counter);
        }
    }

    template <typename T>
    void launch_div_avg(const int* __restrict__ counter, const T* __restrict__ sum, float* __restrict__ avg) {
        div_avg_kernel<<<1, 1>>>(counter, sum, avg);
    }

    __global__ void convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *output = static_cast<float>(*input);
        }
    }

    void launch_convert_double_to_float_kernel(const double* __restrict__ input, float* __restrict__ output) {
        convert_double_to_float_kernel<<<1, 1>>>(input, output);
    }


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
    __device__ void inline atomicReduction(double* addr, double& val, SumOperator<double> &op) {
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


    template <typename T, typename T_res, typename Op>
    __global__ void reduction_sum_wrap(T* __restrict__ arr, T_res* __restrict__ res, char* __restrict__ isnull, const uint32_t sz, const T_res inital) {
        __shared__ T_res wrap_reductions[BLOCK_DIM/32];
        __shared__ T_res sh_res;
        Op op;
        if(threadIdx.x == 0)  
            sh_res = inital;
        // get info
        int tid = threadIdx.x;
        int wrapIdx = tid/32;
        int laneIdx = tid%32;

        T_res localVal = inital;
        int globalIdx = threadIdx.x + COARSENING_FACTOR * blockIdx.x * blockDim.x;
        #pragma unroll
        for(uint32_t k = 0;k < COARSENING_FACTOR;k++) {
            // 1. get local value for each ele in wrap
            localVal = globalIdx < sz && !isnull[globalIdx] ? static_cast<T_res>(arr[globalIdx]) : inital;
            // 2. each wrap reduce
            #pragma unroll
            for(uint32_t d = 16; d > 0; d>>=1){
                T_res neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
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
                    T_res neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                    localVal = op.apply(localVal,  neighbor );
                }
                if(laneIdx == 0) 
                    sh_res = op.apply(localVal, sh_res);
            }
            __syncthreads();
            globalIdx += blockDim.x;
        }
        if(threadIdx.x == 0) atomicReduction(res, sh_res, op); 
    }

    // =============================== for count non nulls =============================
    __global__ void reduction_count_notnull(char* __restrict__ arr, int* __restrict__ res, char* __restrict__ isnull, const uint32_t sz, const int inital) {
        __shared__ int wrap_reductions[BLOCK_DIM/32];
        __shared__ int sh_res;
        SumOperator<int> op;
        if(threadIdx.x == 0)  
            sh_res = inital;
        // get info
        int tid = threadIdx.x;
        int wrapIdx = tid/32;
        int laneIdx = tid%32;

        int localVal = inital;
        int globalIdx = threadIdx.x + COARSENING_FACTOR * blockIdx.x * blockDim.x;
        #pragma unroll
        for(uint32_t k = 0;k < COARSENING_FACTOR;k++) {
            // 1. get local value for each ele in wrap
            localVal = globalIdx < sz ? !isnull[globalIdx] : inital;
            // 2. each wrap reduce
            #pragma unroll
            for(uint32_t d = 16; d > 0; d>>=1){
                int neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                localVal = localVal +  neighbor;
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
                    int neighbor =  __shfl_down_sync(0xFFFFFFFF, localVal, d) ;
                    localVal = localVal +  neighbor;
                }
                if(laneIdx == 0) 
                    sh_res = localVal + sh_res;
            }
            __syncthreads();
            globalIdx += blockDim.x;
        }
        if(threadIdx.x == 0) atomicReduction(res, sh_res, op); 
    }

    template <typename T, typename T_res, typename Op>
    void launch_reduction_operators(T* __restrict__ d_arr, T_res* __restrict__ res, char* __restrict__ isnull,  const uint32_t sz, cudaStream_t& stream, const T_res inital) {
        dim3 threads(BLOCK_DIM);
        dim3 blocks(CEIL_DIV(sz, COARSENING_FACTOR*BLOCK_DIM));

        reduction_sum_wrap<T, T_res, Op><<<blocks, threads, 0, stream>>>(d_arr, res, isnull, sz, inital);
        CUDA_CHECK_LAST();

    }

    void launch_reduction_count_notnull(char* __restrict__ d_arr, int* __restrict__ res, char* __restrict__ isnull,  const uint32_t sz, cudaStream_t& stream, const int inital) {
        dim3 threads(BLOCK_DIM);
        dim3 blocks(CEIL_DIV(sz, COARSENING_FACTOR*BLOCK_DIM));

        reduction_count_notnull<<<blocks, threads, 0, stream>>>(d_arr, res, isnull, sz, inital);
        CUDA_CHECK_LAST();

    }

    template void launch_reduction_operators<int, int, MaxOperator<int>>(int* __restrict__, int* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int);
    template void launch_reduction_operators<int, int, MinOperator<int>>(int* __restrict__, int* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int);
    template void launch_reduction_operators<int, int, SumOperator<int>>(int* __restrict__, int* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int);

    template void launch_reduction_operators<float, float, MaxOperator<float>>(float* __restrict__, float* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const float);
    template void launch_reduction_operators<float, float, MinOperator<float>>(float* __restrict__, float* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const float);
    template void launch_reduction_operators<float, float, SumOperator<float>>(float* __restrict__, float* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const float);
    // allow more percesion in sum
    template void launch_reduction_operators<float, double, SumOperator<double>>(float* __restrict__, double* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const double);
    
    template void launch_reduction_operators<int64_t, int64_t, MaxOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int64_t);
    template void launch_reduction_operators<int64_t, int64_t, MinOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int64_t);
    template void launch_reduction_operators<int64_t, int64_t, SumOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&, const int64_t);
    
    //
    template void launch_div_avg<int>(const int* __restrict__ counter, const int* __restrict__ sum, float* __restrict__ avg);
    template void launch_div_avg<float>(const int* __restrict__ counter, const float* __restrict__ sum, float* __restrict__ avg);
    template void launch_div_avg<double>(const int* __restrict__ counter, const double* __restrict__ sum, float* __restrict__ avg);
    template void launch_div_avg<int64_t>(const int* __restrict__ counter, const int64_t* __restrict__ sum, float* __restrict__ avg);

}