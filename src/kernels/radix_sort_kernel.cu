#include "kernels/radix_sort_kernel.hpp"
#include "kernels/string_kernel.hpp"
#include "kernels/prefix_sum.hpp"
#include "utils/macros.hpp"


namespace YallaSQL::Kernel {
template <size_t BlockDim>
__device__ bool scan_inefficent(int sh_mem[2][BlockDim+1], const int actualBlockSz) {
    bool buff_idx = 0; // buffer
    const uint32_t thidx = threadIdx.x + 1;
    for(uint32_t offset = 1; offset <= actualBlockSz; offset <<= 1) {
        if(thidx >= offset) {
            sh_mem[!buff_idx][thidx] = sh_mem[buff_idx][thidx] + sh_mem[buff_idx][thidx - offset];
        } else {
            sh_mem[!buff_idx][thidx] = sh_mem[buff_idx][thidx];
        }
        buff_idx = !buff_idx;
        __syncthreads();
    }
    return buff_idx;
}

template <typename OP, typename T, size_t BlockDim>
__device__ void one_bit_sort(T sh_A[BlockDim], T res[BlockDim], const int actualBlockSz, const uint32_t N, uint32_t old_idxs[BlockDim], uint32_t new_idxs[BlockDim], int sh_mem[2][BlockDim+1], const uint32_t bit, OP& op) {
    // 1. count
    if(threadIdx.x == 0)
        sh_mem[0][0] = 0;
    if(threadIdx.x < actualBlockSz){
        sh_mem[0][threadIdx.x+1] = op.get_bit(sh_A[threadIdx.x], bit);
        // printf("input of bit %i thix %i A=%i val_of_bit=%i\n", bit,  threadIdx.x, sh_A[threadIdx.x], sh_mem[0][threadIdx.x+1] );
    }
    __syncthreads();
    // 2. scan
    bool buff_idx = scan_inefficent<BlockDim>(sh_mem, actualBlockSz);
    // 3. gather
    // for zeros it's my index - num of ones left me
    // for one it's (total_size - one in total + ones on left)
    if(threadIdx.x < actualBlockSz) { 
        int num_ones = sh_mem[buff_idx][actualBlockSz];
        int ones_left = sh_mem[buff_idx][threadIdx.x];
    
        int new_idx = op.get_bit(sh_A[threadIdx.x], bit) ? 
                    (actualBlockSz - num_ones + ones_left) : // For 1s: place after all 0s
                    (threadIdx.x - ones_left);
        res[new_idx] = sh_A[threadIdx.x];
        new_idxs[new_idx] = old_idxs[threadIdx.x];
    }
    __syncthreads();
}
// global_counter[radix][blockIdx.x]
template <typename OP, typename T>
__device__ void update_glob_buckets(T* A, uint32_t* local_counter, const uint32_t iter, const uint32_t N, const uint32_t global_idx, uint32_t* global_counter, OP& op) {
    if(global_idx < N) {
        int radix = op.get_radix(A[threadIdx.x], iter);
        atomicAdd(&local_counter[radix], 1);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < BUCKET_SZ; i+=blockDim.x) {
        global_counter[GET_IDX(i, blockIdx.x, gridDim.x)] = local_counter[i];
    }
}

template <typename OP, typename T, size_t BlockDim>
__global__ void radix_sort_local_kerenl(T *A, T *res, uint32_t* old_idxs, uint32_t *new_idxs, uint32_t* global_counter, const uint32_t N, const uint32_t iter) {
    OP op;
    int actualBlockSz = (blockIdx.x == gridDim.x - 1)? N - blockIdx.x*BlockDim : BlockDim;

    __shared__ T sh_A[2][BlockDim];
    __shared__ int sh_mem[2][BlockDim+1]; // 2 blocks swaped every time (+1 since it's exclusive sum)
    __shared__ uint32_t sh_idxs[2][BlockDim];
    __shared__ uint32_t local_counter[BUCKET_SZ];
    // 1. get data into local memory
    if(threadIdx.x == 0) {
        sh_mem[0][0] = 0;
        sh_mem[1][0] = 0;
    }
    for(int i = threadIdx.x; i < BUCKET_SZ; i += blockDim.x) {
        local_counter[threadIdx.x] = 0;
    }
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(global_idx < N)
        sh_A[0][threadIdx.x] = A[global_idx];
    // else
        // sh_A[0][threadIdx.x] = 0;
    sh_idxs[0][threadIdx.x] = (global_idx < N) ? old_idxs[global_idx] : 0;
    __syncthreads();
    // 2. perform one bit radix
    bool input_idx = 0;
    #pragma unroll
    for(int r = 0;r < OUR_RADIX;r++) {
        one_bit_sort<OP, T, BlockDim>(sh_A[input_idx], sh_A[!input_idx], actualBlockSz, N, sh_idxs[input_idx], sh_idxs[!input_idx], sh_mem, iter*OUR_RADIX + r, op);
        input_idx = !input_idx;
        __syncthreads();
    } 
    // 3. write it ^ write on global counter
    update_glob_buckets(sh_A[input_idx], local_counter, iter, N, global_idx, global_counter, op);
    if(global_idx < N) {
        res[global_idx] = sh_A[input_idx][threadIdx.x];
        new_idxs[global_idx] = sh_idxs[input_idx][threadIdx.x];
    }
}

// global_counter[radix][blockIdx.x]
template <typename OP, typename T>
__global__ void radix_sort_shuffle(T* A, T* res, uint32_t* old_idxs, uint32_t* new_idxs, uint32_t* global_counter, uint32_t* global_counter_sum, const uint32_t N, const uint32_t iter) {
    OP op;
    __shared__ uint32_t local_counter[BUCKET_SZ];
    for(int i = threadIdx.x; i < BUCKET_SZ; i+= blockDim.x) {
        local_counter[i] = global_counter[GET_IDX(i, blockIdx.x, gridDim.x)];
    }
    __syncthreads();
    // sort locally
    if(threadIdx.x == 0) {
        for(int i = 1; i < BUCKET_SZ;i++) {
            // printf("LOCAL COUNTER %i %i\n", i, local_counter[i]);
            local_counter[i] += local_counter[i - 1];
        }
    }
    __syncthreads();
    //
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx < N) {
        int radix = op.get_radix(A[global_idx], iter);
        int bucket_idx = radix == 0 ? threadIdx.x : threadIdx.x - local_counter[radix - 1];
        int counter_idx = GET_IDX(radix, blockIdx.x, gridDim.x) - 1;
        int sum = counter_idx >= 0 ? global_counter_sum[counter_idx] : 0;
        int new_idx = bucket_idx + sum;
        // printf("blcokIdx %i radix %i bucket_idx %i sum %i newidx %i\n",blockIdx.x, radix, bucket_idx, sum, new_idx);
        // printf("old: %i new: %i\n", global_idx, new_idx);
        // printf("%i %i %i %i %i\n", radix, bucket_idx, counter_idx, sum, new_idx);
        res[new_idx] = A[global_idx];
        new_idxs[new_idx] = old_idxs[global_idx];
    }
}

// will show results inside d_input, d_idxs_in
template <typename OP, typename T, size_t BlockDim>
void launch_radix_sort(T* d_input, uint32_t* &d_idxs_in, const uint32_t N, cudaStream_t stream) {
    T *d_output;
    uint32_t *d_idxs_out;
    CUDA_CHECK(cudaMallocAsync(&d_output, N * sizeof(T), stream));
    CUDA_CHECK(cudaMallocAsync(&d_idxs_in, N * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_idxs_out, N * sizeof(uint32_t), stream));
    // set first as input
    // storing indexs for argsort in rest of columns
    uint32_t idxs_in[N]; 
    for(int i = 0;i < N;i++) idxs_in[i] = i;
    CUDA_CHECK(cudaMemcpyAsync(d_idxs_in, idxs_in, N * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    // initalize global counters
    // 1. init global counter
    uint32_t* global_counter, *global_counter_sum;
    dim3 blocks( CEIL_DIV(N, BlockDim) );
    uint32_t counter_sz = blocks.x * BUCKET_SZ;
    // for global counter
    CUDA_CHECK( cudaMallocAsync((void**)&global_counter, counter_sz * sizeof(uint32_t), stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&global_counter_sum, counter_sz * sizeof(uint32_t), stream) );//+1 to handle exclusive
    // =================== ITEREATE ===================
    const int num_iters = (sizeof(T)*8)/OUR_RADIX;
    for(int iter = 0;iter < num_iters;++iter) {
        radix_sort_local_kerenl<OP, T, BlockDim><<<blocks, BlockDim, 0, stream>>>(d_input, d_output, d_idxs_in, d_idxs_out, global_counter, N, iter);
        // prefix sum on global counter
        launch_prefix_sum(global_counter, global_counter_sum, counter_sz, stream);
        // update/global shuffling
        radix_sort_shuffle<OP, T><<<blocks, BlockDim, 0, stream>>>(d_output, d_input, d_idxs_out, d_idxs_in, global_counter, global_counter_sum, N, iter);
    }


}
    // Explicit template instantiations
    template void launch_radix_sort<AscOp<uint32_t>, uint32_t, BLOCK_DIM>(uint32_t*, uint32_t*&,const uint32_t, cudaStream_t);
    template void launch_radix_sort<AscOp<uint64_t>, uint64_t, BLOCK_DIM>(uint64_t*, uint32_t*&,const uint32_t, cudaStream_t);
    template void launch_radix_sort<AscOpString, String, BLOCK_DIM_STR>(String*, uint32_t*&, const uint32_t, cudaStream_t);

    template void launch_radix_sort<DescOp<uint32_t>, uint32_t, BLOCK_DIM>(uint32_t*, uint32_t*&,const uint32_t, cudaStream_t);
    template void launch_radix_sort<DescOp<uint64_t>, uint64_t, BLOCK_DIM>(uint64_t*, uint32_t*&,const uint32_t, cudaStream_t);
    template void launch_radix_sort<DescOpString, String, BLOCK_DIM_STR>(String*, uint32_t*&, const uint32_t, cudaStream_t);
}
