#include "kernels/constants.hpp"
#include "kernels/batch_wrapper.hpp"
// reference https://www.youtube.com/watch?v=szoc52lNufU&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=14
// we can use COARSING_FACTOR  as ELE_PER_THREADS
// number of out elements per thread
#define ELE_PER_THREADS 6
// number of out elements per block
#define ELE_PER_BLOCK ELE_PER_THREADS*BLOCK_DIM

// do binary search to get (i) where is coRank in "T* a"
__device__ unsigned int _find_co_rank(BatchWrapper* a, BatchWrapper* b, unsigned int m, unsigned int n, unsigned int k) {
    unsigned int l = k > n ? k - n : 0; // left most || lower bound max(k - n, 0)
    unsigned int r = k > m ? m : k; // right most || upper bound min(k, m)
    unsigned int i = (l + r) / 2;

    while(l <= r) {
        if(i == m) return i;
        // if(i < m && k - i > 0  && a[i] < b[k - i - 1]) // i is too low
        // else if(i > 0 && k - i  < n && b[k - i] < a[i - 1]) // i is too high
        if(i < m && k - i > 0  && a->less_than(b, i, k - i - 1)) // i is too low
            l = i;
        else if(i > 0 && k - i  < n && b->less_equal(a, k - i, i - 1)) // i is too high
            r = i;
        else
            return i;

        i = (l + r) / 2;
    }

    return i;
}

__global__ void merge_kerenel(BatchWrapper* a, BatchWrapper* b, BatchWrapper* res, unsigned int m, unsigned int n) {
    unsigned int k = (threadIdx.x + blockIdx.x * blockDim.x)* ELE_PER_THREADS;
    if(k < m + n) {
        unsigned int i = _find_co_rank(a, b, m, n, k);
        unsigned int j =  k - i;

        #pragma unroll
        for(unsigned int d = 0; d < ELE_PER_BLOCK; d++) {
            // if(j >= n || a[i] <= b[i]) 
            if(j >= n || a->less_equal(b, i, j)) 
                res[k+d] = a[i++];
            else 
                res[k+d] = b[j++];
        }

    }
}

// coleased access version
__global__ void merge_kerenel_v2(BatchWrapper* a, BatchWrapper* b, BatchWrapper* res, unsigned int m, unsigned int n) {
    __shared__ BatchWrapper sh_a;
    __shared__ BatchWrapper sh_b;
    __shared__ BatchWrapper sh_res;

    unsigned int start_k_block, last_k_block, start_i_block, last_i_block, start_j_block, last_j_block;
    if(threadIdx.x == 0) {
        // initalize batch_wrapper
        sh_a.bytes = a->bytes;
        sh_a.ncol = a->ncol;
        
        sh_b.bytes = b->bytes;
        sh_b.ncol = b->ncol;

        sh_res.bytes = b->bytes;
        sh_res.ncol = b->ncol;
        
        // get starter of block
        start_k_block = blockIdx.x * ELE_PER_BLOCK;
        last_k_block = blockIdx.x == gridDim.x - 1 ? m+n : start_k_block + ELE_PER_BLOCK;
        // two binary loops to get first i & last i
        start_i_block = _find_co_rank(a, b, m, n, start_k_block); 
        start_j_block = start_k_block - start_i_block;
        
        last_i_block = _find_co_rank(a, b, m, n, last_k_block);
        last_j_block = last_k_block - last_i_block;

        sh_a.batchSize = last_i_block - start_i_block;
        sh_b.batchSize = last_j_block - start_j_block;
        sh_res.batchSize = last_k_block - start_k_block;
    }
    __syncthreads();
    // copy shared data
    for(unsigned int col = 0; col < sh_a.ncol;col++) { // copy col by col
       for(unsigned int ii = threadIdx.x; ii < sh_a.batchSize; ii += blockDim.x) { 
            sh_a.copy_cell(a, start_i_block + ii, ii, col);//dist.copy_cell(src, src_idx, dist_idx, col)
        }
        for(unsigned int jj = threadIdx.x; jj < sh_b.batchSize; jj += blockDim.x) { 
            sh_b.copy_cell(b, start_j_block + jj, jj, col);
        }
    }
    __syncthreads();
    // thread merge block 
    unsigned int k = threadIdx.x * ELE_PER_THREADS;
    if(k < sh_res.batchSize) {
        unsigned int i = _find_co_rank(&sh_a, &sh_b, sh_a.batchSize, sh_b.batchSize, k);
        unsigned int j =  k - i;

        #pragma unroll
        for(unsigned int d = 0; d < ELE_PER_BLOCK && k + d < m + n; d++) {
            for(unsigned int col = 0; col < sh_a.ncol;col++) {
                if(j >= n || a->less_equal(b, i, j)) 
                    sh_res.copy_cell(&sh_a, i++, k+d, col);
                else 
                    sh_res.copy_cell(&sh_b, j++, k+d, col);
            }
        }

    }

    // res[] = sh_res[]
}