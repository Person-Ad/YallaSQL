#include "kernels/comparison_operators_kernel.hpp"

namespace YallaSQL::Kernel {
    template <typename T, typename Op>
    __global__ void apply_batches(T* __restrict__ rhs, T* __restrict__ lhs, bool* res, const unsigned int sz, bool isneg) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = isneg ? !op.apply(lhs[global_idx], rhs[global_idx]) : op.apply(lhs[global_idx], rhs[global_idx]);
            }
        }

    }


    template <typename T, typename Op>
    __global__ void apply_batches_scalar_rhs(T* __restrict__ rhs, T* __restrict__ lhs, bool* res, const unsigned int sz, bool isneg) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = isneg ? !op.apply(lhs[global_idx], rhs[0]) : op.apply(lhs[global_idx], rhs[0]);
            }
        }

    }

    template <typename T, typename Op>
    __global__ void apply_batches_scalar_lhs(T* __restrict__ rhs, T* __restrict__ lhs, bool* res, const unsigned int sz, bool isneg) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = isneg ? !op.apply(lhs[0], rhs[global_idx]) : op.apply(lhs[0], rhs[global_idx]);
            }
        }

    }
    


    template <typename T, typename Op>
    void launch_conditional_operators(T* __restrict__ d_rhs, T* __restrict__ d_lhs, OperandType t_rhs, OperandType t_lhs, bool* d_res, unsigned int sz, cudaStream_t& stream, bool isneg) {
        if(t_rhs == OperandType::SCALAR && t_lhs == OperandType::SCALAR) {
            apply_batches<T, Op><<<1, 1>>>(d_rhs, d_lhs, d_res, sz, isneg);
            return;
        } 
        
        dim3 threads(BLOCK_DIM);
        dim3 blocks (CEIL_DIV(sz, threads.x * COARSENING_FACTOR));
        
        if(t_rhs == OperandType::SCALAR) 
            apply_batches_scalar_rhs<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz, isneg);
        else if(t_lhs == OperandType::SCALAR)
            apply_batches_scalar_lhs<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz, isneg);
        else
            apply_batches<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz, isneg);

        CUDA_CHECK_LAST();
    }

   // Explicit instantiations with __restrict__
    template void launch_conditional_operators<int, LEOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int, GEOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int, LEQOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int, GEQOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int, EQOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int, NEQOperator<int>>(int* __restrict__, int* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);

    template void launch_conditional_operators<float, LEOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<float, GEOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<float, LEQOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<float, GEQOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<float, EQOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<float, NEQOperator<float>>(float* __restrict__, float* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);

    template void launch_conditional_operators<int64_t, LEOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int64_t, GEOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int64_t, LEQOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int64_t, GEQOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int64_t, EQOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<int64_t, NEQOperator<int64_t>>(int64_t* __restrict__, int64_t* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);

    template void launch_conditional_operators<String, LEOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<String, GEOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<String, LEQOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<String, GEQOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<String, EQOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<String, NEQOperator<String>>(String* __restrict__, String* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);


    template void launch_conditional_operators<bool, ANDOperator>(bool* __restrict__, bool* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);
    template void launch_conditional_operators<bool, OROperator>(bool* __restrict__, bool* __restrict__, OperandType, OperandType, bool*, unsigned int, cudaStream_t&, bool);

}