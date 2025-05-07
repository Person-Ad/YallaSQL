#include "kernels/prefix_sum.hpp"
#include "kernels/constants.hpp"
#include "utils/macros.hpp"

namespace YallaSQL::Kernel {
#define PREFIX_SH_MEM_SZ (2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS)
template <typename T>
__device__ void _add_prev_block(T* res, T* sh_mem, 
                                        const int &actualBlockIdx, 
                                        const uint32_t &sz, 
                                        const uint32_t &arr_idx,
                                        uint32_t& blocks_finished) {
    __syncthreads();
    // wait for previous
    __shared__ T prev_sum;
    while(atomicAdd(&blocks_finished, 0) != actualBlockIdx) {__nanosleep(100);}
    if(threadIdx.x == 0) {
        prev_sum = actualBlockIdx == 0 ? 0 : 
                   (2 * blockDim.x * actualBlockIdx - 1 < sz ? res[2 * blockDim.x * actualBlockIdx - 1] : 0);
        // prev_sum = actualBlockIdx == 0 ? 0 : res[2*blockDim.x - 1 + 2 * blockDim.x * (actualBlockIdx - 1)];
    }
    __syncthreads();
    // let's go & sum previous
    // 3. write output
    int ai = threadIdx.x; ai += CONFLICT_FREE_OFFSET(ai);
    int bi = threadIdx.x + blockDim.x; bi += CONFLICT_FREE_OFFSET(bi);
    // write once in result
    if(arr_idx < sz)
        res[arr_idx] = sh_mem[ai] + prev_sum;
    if(arr_idx + blockDim.x < sz)
        res[arr_idx + blockDim.x] = sh_mem[bi] + prev_sum;

    __syncthreads();
    if(threadIdx.x == blockDim.x - 1) {
        __threadfence();
        atomicAdd(&blocks_finished, 1);
    }    
}

template <typename T>
__device__ void upsweep_phase(T* sh_mem, uint64_t &offset) {
    int ai, bi;
    offset = 1;
    #pragma unroll
    for(uint32_t d = BLOCK_DIM; d > 0; d>>= 1) {
        ai = 2*offset*(threadIdx.x + 1) - 1; ai += CONFLICT_FREE_OFFSET(ai);
        bi = 2*offset*(threadIdx.x + 1) - offset - 1; bi += CONFLICT_FREE_OFFSET(bi);
        if(threadIdx.x < d) {
            sh_mem[ai] += sh_mem[bi];
        }
        offset <<= 1;
        __syncthreads();
    }
}

template <typename T>
__device__ void downsweep_phase(T* sh_mem, uint64_t &offset) {
    int ai, bi;
    #pragma unroll
    for(uint32_t d = 1; d <= BLOCK_DIM; d<<= 1) {
        offset >>= 1;
        ai = 2*offset*(threadIdx.x + 1) + offset - 1;  ai += CONFLICT_FREE_OFFSET(ai);
        bi = 2*offset*(threadIdx.x + 1) - 1;  bi += CONFLICT_FREE_OFFSET(bi);
        if(ai < 2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS)  {
            sh_mem[ai] += sh_mem[bi];
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void prefix_sum(T* arr, T* res, 
                            const uint32_t sz,
                            uint32_t& blocks_counter,   // number should i take
                            uint32_t& blocks_finished) { // number of blocks finished
    __shared__ int actualBlockIdx;
    __shared__ T sh_mem[2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS]; // add padding to sh_mem
    for(int k = 0;k < COARSENING_FACTOR; k++) {
        // 1. get actual blockIdx
        if(threadIdx.x == 0) {
            actualBlockIdx = atomicAdd(&blocks_counter, 1);
        }
        __syncthreads(); // ensure all threads see updated

        const int arr_idx = threadIdx.x + 2 * blockDim.x * actualBlockIdx; // 
        const int th = threadIdx.x;
        // 1. get data into sh_mem
        int ai = th; ai += CONFLICT_FREE_OFFSET(ai);
        int bi = th + blockDim.x; bi += CONFLICT_FREE_OFFSET(bi);
        sh_mem[ai] = arr_idx >= sz ? 0 : arr[arr_idx]; // [arr[0], arr[1], arr[2], arr[3]]
        sh_mem[bi] = arr_idx  + blockDim.x >= sz ? 0 : arr[arr_idx  + blockDim.x];
        __syncthreads();
        // 2. upsweep
        uint64_t offset = 1;
        upsweep_phase(sh_mem, offset);
        // 3. downsweep
        downsweep_phase(sh_mem, offset);
        
        _add_prev_block(res, sh_mem, actualBlockIdx, sz, arr_idx, blocks_finished);
        __syncthreads();
    }

}

// to help in filters
__global__ void prefix_sum_mask(bool* arr, uint32_t* res, 
                            const uint32_t sz,
                            uint32_t& blocks_counter,   // number should i take
                            uint32_t& blocks_finished) { // number of blocks finished
    __shared__ int actualBlockIdx;
    __shared__ uint32_t sh_mem[PREFIX_SH_MEM_SZ]; // add padding to sh_mem
    for(int k = 0;k < COARSENING_FACTOR; k++) {
        // 1. get actual blockIdx
        if(threadIdx.x == 0) {
            actualBlockIdx = atomicAdd(&blocks_counter, 1);
        }
        __syncthreads(); // ensure all threads see updated

        const int arr_idx = threadIdx.x + 2 * blockDim.x * actualBlockIdx; // 
        const int th = threadIdx.x;
        // 1. get data into sh_mem
        int ai = th; ai += CONFLICT_FREE_OFFSET(ai);
        int bi = th + blockDim.x; bi += CONFLICT_FREE_OFFSET(bi);

        sh_mem[ai] = arr_idx >= sz ? 0 : static_cast<uint32_t>(arr[arr_idx]); // [arr[0], arr[1], arr[2], arr[3]]
        sh_mem[bi] = arr_idx  + blockDim.x >= sz ? 0 : static_cast<uint32_t>(arr[arr_idx  + blockDim.x]);
        __syncthreads();
        // 2. upsweep
        uint64_t offset = 1;
        upsweep_phase(sh_mem, offset);
        // 3. downsweep
        downsweep_phase(sh_mem, offset);
        
        _add_prev_block(res, sh_mem, actualBlockIdx, sz, arr_idx, blocks_finished);
        __syncthreads();
    }
}


void launch_prefix_sum_mask(bool* arr, uint32_t* res, const uint32_t sz, cudaStream_t &stream) {
    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(sz, 2 * threads.x * COARSENING_FACTOR));
     
    uint32_t *blocks_counter, *blocks_finished;
    CUDA_CHECK( cudaMallocAsync((void**)&blocks_counter, sizeof(uint32_t), stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&blocks_finished, sizeof(uint32_t), stream) );

    CUDA_CHECK( cudaMemsetAsync(blocks_counter, 0, sizeof(uint32_t), stream) );
    CUDA_CHECK( cudaMemsetAsync(blocks_finished, 0, sizeof(uint32_t), stream) );

    prefix_sum_mask<<<blocks, threads, 0, stream>>>(arr, res, sz, *blocks_counter, *blocks_finished);
    
    CUDA_CHECK( cudaFreeAsync(blocks_counter, stream) );
    CUDA_CHECK( cudaFreeAsync(blocks_finished, stream) );
    CUDA_CHECK_LAST();
}

template <typename T>
void launch_prefix_sum(T* arr, T* res, const uint32_t sz, cudaStream_t &stream) {
    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(sz, 2 * threads.x * COARSENING_FACTOR));
    
    uint32_t *blocks_counter, *blocks_finished;
    CUDA_CHECK( cudaMallocAsync((void**)&blocks_counter, sizeof(uint32_t), stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&blocks_finished, sizeof(uint32_t), stream) );

    CUDA_CHECK( cudaMemsetAsync(blocks_counter, 0, sizeof(uint32_t), stream) );
    CUDA_CHECK( cudaMemsetAsync(blocks_finished, 0, sizeof(uint32_t), stream) );

    prefix_sum<<<blocks, threads, 0, stream>>>(arr, res, sz, *blocks_counter, *blocks_finished);
    
    CUDA_CHECK( cudaFreeAsync(blocks_counter, stream) );
    CUDA_CHECK( cudaFreeAsync(blocks_finished, stream) );
    CUDA_CHECK_LAST();
}

    template void launch_prefix_sum<uint32_t>(uint32_t* arr, uint32_t* res, const uint32_t sz, cudaStream_t &stream);

}