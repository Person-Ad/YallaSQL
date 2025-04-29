#include "kernels/binary_operators_kernel.hpp"

namespace YallaSQL::Kernel
{
    // batch * batch || scalar * scalar
    template <typename T, typename Op>
    __global__ void apply_batches(T* rhs, T* lhs, T* res, const unsigned int sz) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = op.apply(lhs[global_idx], rhs[global_idx]);
            }
        }
    }
    // batch * scalar
    template <typename T, typename Op>
    __global__ void apply_batches_scalar_rhs(T* rhs, T* lhs, T* res, const unsigned int sz) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = op.apply(lhs[global_idx], rhs[0]);
            }
        }
    }
    // scalar * batch
    template <typename T, typename Op>
    __global__ void apply_batches_scalar_lhs(T* rhs, T* lhs, T* res, const unsigned int sz) {
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
        unsigned int stride = blockDim.x * gridDim.x;
    
        Op op;

        #pragma unroll 
        for (int k = 0; k < COARSENING_FACTOR; k++) {
            unsigned int global_idx = idx + k * stride;
            if (global_idx < sz) {
                res[global_idx] = op.apply(lhs[0], rhs[global_idx]);
            }
        }
    }
    
    // TODO: add streaming
    template <typename T, typename Op>
    void launch_binary_operators(T* d_rhs, T* d_lhs, OperandType t_rhs, OperandType t_lhs, T* d_res, unsigned int sz, cudaStream_t& stream) {
        if(t_rhs == OperandType::SCALAR && t_lhs == OperandType::SCALAR) {
            apply_batches<T, Op><<<1, 1>>>(d_rhs, d_lhs, d_res, sz);
            return;
        } 
        
        dim3 threads(BLOCK_DIM);
        dim3 blocks (CEIL_DIV(sz, threads.x * COARSENING_FACTOR));
        
        if(t_rhs == OperandType::SCALAR) 
            apply_batches_scalar_rhs<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz);
        else if(t_lhs == OperandType::SCALAR)
            apply_batches_scalar_lhs<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz);
        else
            apply_batches<T, Op><<<blocks, threads, 0, stream>>>(d_rhs, d_lhs, d_res, sz);

        CUDA_CHECK_LAST();
    }

    // Explicit template instantiations
    template void launch_binary_operators<int, AddOperator<int>>(int*, int*, OperandType, OperandType, int*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int, MinusOperator<int>>(int*, int*, OperandType, OperandType, int*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int, MulOperator<int>>(int*, int*, OperandType, OperandType, int*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int, DivOperator<int>>(int*, int*, OperandType, OperandType, int*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int, RemOperator<int>>(int*, int*, OperandType, OperandType, int*, unsigned int, cudaStream_t&);

    template void launch_binary_operators<int64_t, AddOperator<int64_t>>(int64_t*, int64_t*, OperandType, OperandType, int64_t*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int64_t, MinusOperator<int64_t>>(int64_t*, int64_t*, OperandType, OperandType, int64_t*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int64_t, MulOperator<int64_t>>(int64_t*, int64_t*, OperandType, OperandType, int64_t*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int64_t, DivOperator<int64_t>>(int64_t*, int64_t*, OperandType, OperandType, int64_t*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<int64_t, RemOperator<int64_t>>(int64_t*, int64_t*, OperandType, OperandType, int64_t*, unsigned int, cudaStream_t&);


    template void launch_binary_operators<float, AddOperator<float>>(float*, float*, OperandType, OperandType, float*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<float, MinusOperator<float>>(float*, float*, OperandType, OperandType, float*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<float, MulOperator<float>>(float*, float*, OperandType, OperandType, float*, unsigned int, cudaStream_t&);
    template void launch_binary_operators<float, DivOperator<float>>(float*, float*, OperandType, OperandType, float*, unsigned int, cudaStream_t&);

} // namespace YallaSQL::GPU