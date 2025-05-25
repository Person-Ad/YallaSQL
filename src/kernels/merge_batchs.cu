#include "kernels/string_kernel.hpp"
#include "kernels/merge_batchs.hpp"
#include "kernels/constants.hpp"
#include "utils/macros.hpp"
namespace YallaSQL::Kernel
{

#define ELE_PER_TH 6
#define ELE_PER_BLOCK (6*256)



template <typename T, typename OP>
__host__ __device__ int find_corank(T* A, T* B, uint32_t m, uint32_t n, uint32_t k) {
    OP op;
    uint32_t l = k > n ? k - n : 0; 
    uint32_t r = k < m ? k : m; 
    uint32_t i, j;
    while(l <= r) {
        i = (l + r) / 2;
        j = k - i;
        if(j > 0 && i < m  && op.is_greater(B[j - 1], A[i]))
            l = i + 1;
        else if(i > 0 && j < n &&  op.is_greater(A[i - 1],B[j]))
            r = i - 1;
        else
            return i;
    }
    return l;
}

template <typename T, typename OP>
__global__ void merge_sorted_array_kernel_v1(T* A, T* B, T* C, uint32_t* newIdxs, bool* tableIdxs, int* lasti, const uint32_t k_mx, uint32_t m, uint32_t n) {
   OP op;
    uint32_t k = ELE_PER_TH * (threadIdx.x + blockIdx.x * blockDim.x);
    
    // Find the last position from A in the merged array
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *lasti = find_corank<T, OP>(A, B, m, n, k_mx);
    }
    
    if (k < n + m && k < k_mx) {
        uint32_t i = find_corank<T, OP>(A, B, m, n, k);
        uint32_t j = k - i;
        
        for (int d = 0; d < ELE_PER_TH && k + d < k_mx && k + d < n + m; d++) {
            if (j >= n) {
                // Take from A
                C[k + d] = A[i];
                tableIdxs[k + d] = true;  // From table A
                newIdxs[k + d] = i;
                i++;
            }
            else if (i >= m) {
                // Take from B
                C[k + d] = B[j];
                tableIdxs[k + d] = false;  // From table B
                newIdxs[k + d] = j;
                j++;
            }
            else if (op.is_less_or_equal(A[i], B[j])) {
                // Take from A
                C[k + d] = A[i];
                tableIdxs[k + d] = true;  // From table A
                newIdxs[k + d] = i;
                i++;
            }
            else {
                // Take from B
                C[k + d] = B[j];
                tableIdxs[k + d] = false;  // From table B
                newIdxs[k + d] = j;
                j++;
            }
        }
    }
}
// __global__ void merge_sorted_array_kernel(int* A, int* B, int* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n) {
//     // if(threadIdx.x == 0) {
//     //     printf("Start where m %i, n %i   A[0]=%i B[0]=%i\n", m, n, A[0], B[0]);
//     // }
//     __shared__ int A_sh[ELE_PER_BLOCK];
//     __shared__ int B_sh[ELE_PER_BLOCK];
//     __shared__ int C_sh[ELE_PER_BLOCK];
//     __shared__ uint32_t block_start_k, block_next_k;
//     __shared__ uint32_t block_start_i, block_next_i;
//     __shared__ uint32_t block_start_j, block_next_j;
//     if(threadIdx.x == 0) {
//         //TODO: haha
//         if(blockIdx.x == 0){
//             *lasti = find_corank(A, B, m, n, k_mx);
//         }
//         block_start_k = ELE_PER_TH * (blockIdx.x * blockDim.x);
//         block_next_k  = block_start_k + ELE_PER_BLOCK > k_mx ? k_mx : block_start_k + ELE_PER_BLOCK ;

//         block_start_i = find_corank(A, B, m, n, block_start_k);
//         block_next_i = find_corank(A, B, m, n, block_next_k);
        
//         block_start_j = block_start_k - block_start_i;
//         block_next_j = block_next_k - block_next_i;
//         if(block_start_i >= block_next_i){
//             for(int i = block_start_i ;i < block_start_i+10;i++) printf("A[%i]=%i\n", i, A[i]);
//             for(int i = block_start_j ;i < block_start_j+10;i++) printf("B[%i]=%i\n", i, B[i]);
//             printf("n = %i m = %i st = %i end = %i BlockIdx %i A[i] %i  A[ni] %i \n", n, m, block_start_i, block_next_i, blockIdx.x, A[block_start_i], A[block_next_i]);
//         }
//     }
//     __syncthreads();
//     uint32_t m_sh = block_next_i - block_start_i;
//     uint32_t n_sh = block_next_j - block_start_j;
//     for(int i = threadIdx.x; i < m_sh; i += blockDim.x) {
//         if(block_start_i + i < ELE_PER_BLOCK)
//         A_sh[i] = A[block_start_i + i];
//     }
//     for(int j = threadIdx.x;  j < n_sh; j += blockDim.x) {
//         B_sh[j] = B[block_start_j + j];
//     }
//     __syncthreads();
    
    
//     uint32_t k = threadIdx.x * ELE_PER_TH;
//     if(k < n_sh + m_sh) {

//         uint32_t i = find_corank(A_sh, B_sh, m_sh, n_sh, k);
//         uint32_t j = k - i;

//         for(int d = 0; d < ELE_PER_TH && k + d < n_sh + m_sh; d++) {
//             if(j >= n_sh) C_sh[k + d] = A_sh[i++];
//             else if(i >= m_sh) C_sh[k + d] = B_sh[j++];
//             else if(A_sh[i] <= B_sh[j]) C_sh[k + d] = A_sh[i++];
//             else C_sh[k + d] = B_sh[j++];
//         }

//     }
//     __syncthreads();
//     for(int i = threadIdx.x; i < block_next_k; i+=blockDim.x){
//         if(block_start_k + i < k_mx)
//             C[block_start_k + i] = C_sh[i];
//         // if(block_start_k + i < 10)
//             // printf("out C[%i]=%i\n", block_start_k + i, C_sh[i]);
//     }
// }

template <typename T, typename OP>
void launch_merge_sorted_array_kernel(T* A, T* B, T* C, uint32_t* new_idx, bool* table_idx, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n, cudaStream_t stream) {
    dim3 blocks(CEIL_DIV(k_mx, ELE_PER_BLOCK));
    // YallaSQL::Kernel::merge_sorted_array_kernel<<<blocks, BLOCK_DIM>>>(A, B, C, lasti, k_mx, m, n);    
    YallaSQL::Kernel::merge_sorted_array_kernel_v1<T, OP><<<blocks, BLOCK_DIM, 0, stream>>>(A, B, C, new_idx, table_idx, lasti, k_mx, m, n);    
}

template void launch_merge_sorted_array_kernel<int, AscSortMergeOp<int>>(int*, int*, int*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<float, AscSortMergeOp<float>>(float*, float*, float*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<int64_t, AscSortMergeOp<int64_t>>(int64_t*, int64_t*, int64_t*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<String, AscSortStrOp>(String*, String*, String*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);

template void launch_merge_sorted_array_kernel<int, DescSortMergeOp<int>>(int*, int*, int*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<float, DescSortMergeOp<float>>(float*, float*, float*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<int64_t, DescSortMergeOp<int64_t>>(int64_t*, int64_t*, int64_t*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);
template void launch_merge_sorted_array_kernel<String, DescSortStrOp>(String*, String*, String*, uint32_t*, bool*, int*, const uint32_t, uint32_t, uint32_t, cudaStream_t);

}