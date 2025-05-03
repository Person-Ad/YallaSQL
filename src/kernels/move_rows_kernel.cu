#include "kernels/move_rows_kernel.hpp"
#include "kernels/string_kernel.hpp"
#include "null_bit_set.hpp"

namespace YallaSQL::Kernel {

//TODO: make it coleased access IDK how
template<typename T>
__global__ void move_rows_filter_kernel(T* __restrict__ src, T* res,
                                    uint32_t* __restrict__ map, // map[oldIdx] = newIdx + 1
                                    bool*    __restrict__ mask,
                                    const uint32_t srcSz) 
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // oldIdx
    const unsigned int stride = blockDim.x * gridDim.x;

    #pragma unroll 
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        const unsigned int global_idx = idx + k * stride;
        if(global_idx < srcSz && mask[global_idx]) {
            res[map[global_idx] - 1] = src[global_idx]; 
        }
    }
}

template<typename T>
void launch_move_rows_filter_kernel(T* __restrict__ src, T* res, uint32_t* __restrict__ map, bool* __restrict__ mask, const uint32_t srcSz, cudaStream_t& stream) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(srcSz, threads.x * COARSENING_FACTOR));

    move_rows_filter_kernel<T><<<blocks, threads, 0, stream>>>(src, res, map, mask, srcSz);

    CUDA_CHECK_LAST();
}

    // Explicit template instantiations
    template void launch_move_rows_filter_kernel<int>(int* __restrict__, int*, uint32_t* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<float>(float* __restrict__, float*, uint32_t* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<int64_t>(int64_t* __restrict__, int64_t*, uint32_t* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<String>(String* __restrict__, String*, uint32_t* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
    // for nullset
    template void launch_move_rows_filter_kernel<char>(char* __restrict__, char*, uint32_t* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
}
