#include <bits/stdint-intn.h>
#include "kernels/cast_operators_kernel.hpp"

namespace YallaSQL::Kernel {

template<typename T_src, typename T_dist>
__global__ void numerical_cast_kernel(T_src* src, T_dist* dist, const size_t sz) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    #pragma unroll 
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        const unsigned int global_idx = idx + k * stride;
        if (global_idx < sz) {
            dist[global_idx] = static_cast<T_dist>(src[global_idx]);
        }
    }
}




template<typename T_src, typename T_dist>
void launch_numerical_cast(T_src* src, T_dist* dist, const unsigned int sz, cudaStream_t& stream) {
    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(sz, threads.x * COARSENING_FACTOR));

    numerical_cast_kernel<T_src, T_dist><<<blocks, threads, 0, stream>>>(src, dist, sz);

    CUDA_CHECK_LAST();
}

    // Explicit template instantiations
    template void launch_numerical_cast<int, float>(int*, float*, unsigned int, cudaStream_t&);
    template void launch_numerical_cast<int, int64_t>(int*, int64_t*, unsigned int, cudaStream_t&);
    
    template void launch_numerical_cast<float, int>(float*, int*, unsigned int, cudaStream_t&);
    template void launch_numerical_cast<float, int64_t>(float*, int64_t*, unsigned int, cudaStream_t&);

    template void launch_numerical_cast<int64_t, int>(int64_t*, int*, unsigned int, cudaStream_t&);
    template void launch_numerical_cast<int64_t, float>(int64_t*, float*, unsigned int, cudaStream_t&);
}