#include "kernels/move_rows_kernel.hpp"
#include "kernels/string_kernel.hpp"

namespace YallaSQL::Kernel {


template<typename T>
__global__ void move_rows_join_kernel(const T* __restrict__ src, T* __restrict__ res,
                                    const uint32_t* __restrict__ pairs, // map[oldIdx] = newIdx + 1
                                    const char* __restrict__ isnull,
                                    char* __restrict__ out_isnull,
                                    const uint32_t batchSize,
                                    const bool isright) 
{
    const unsigned int oidx = (threadIdx.x + blockIdx.x * blockDim.x);
    const unsigned int stride = blockDim.x * gridDim.x;
    
    #pragma unroll 
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        const unsigned int global_idx = oidx + k * stride;
        
        if(global_idx < batchSize) {
            const unsigned int pair_idx = 2 * global_idx; // leftIdx, rightIdx
            const unsigned int idx = pairs[pair_idx + (isright ? 1 : 0)];

            res[global_idx] = src[idx]; 
            out_isnull[global_idx] = isnull[idx];
        }
    }
}
//TODO: make it coleased access IDK how
template<typename T>
__global__ void move_rows_filter_kernel(T* __restrict__ src, T* res,
                                    uint32_t* __restrict__ map, // map[oldIdx] = newIdx + 1
                                    bool* __restrict__ mask,
                                    char* __restrict__ isnull,
                                    const uint32_t srcSz) 
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // oldIdx
    const unsigned int stride = blockDim.x * gridDim.x;

    #pragma unroll 
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        const unsigned int global_idx = idx + k * stride;
        if(global_idx < srcSz && mask[global_idx] && !isnull[global_idx]) {
            res[map[global_idx] - 1] = src[global_idx]; 
        }
    }
}

template<typename T>
__global__ void move_one_col(const T* __restrict__ src, 
                            T* __restrict__  res,
                            const char* __restrict__ src_isnull,
                            char* __restrict__ res_isnull,
                            uint32_t* __restrict__ map, // map[oldIdx] = newIdx
                            const uint32_t srcSz) 
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // oldIdx
    const unsigned int stride = blockDim.x * gridDim.x;

    #pragma unroll 
    for (int k = 0; k < COARSENING_FACTOR; k++) {
        const unsigned int global_idx = idx + k * stride;
        if(global_idx < srcSz) {
            res[global_idx] = src[map[global_idx]]; 
            res_isnull[global_idx] = src_isnull[map[global_idx]]; 
        }
    }
}



//TODO: make it coleased access IDK how
// __global__ void move_rows_all_cols_kernel(const void** __restrict__ src, void** __restrict__ res, 
//                                 const uint32_t* __restrict__ newIdxs, // map[oldIdx] = newIdx + 1
//                                 const DataType* __restrict__ types,
//                                 const uint32_t nrows,
//                                 const uint32_t ncols) 
// {
//     const unsigned int col = blockIdx.y;
//     const unsigned int row = threadIdx.x + blockIdx.x * blockDim.x; // oldIdx
//     const unsigned int stride = blockDim.x * gridDim.x;
//     if(col < ncols) {
//         const char* src_col = static_cast<const char*>(src[col]);
//         char* res_col = static_cast<char*>(res[col]);
//         unsigned int bytes = getDataTypeNumBytes(types[col]);
//         #pragma unroll 
//         for (int k = 0; k < COARSENING_FACTOR; k++) {
//             const unsigned int global_idx = row + k * stride;
//             if(global_idx < nrows) {
//                 const char* src_row = src_col + bytes * global_idx;
//                 char* res_row       = res_col + bytes * newIdxs[global_idx];
//                 for(int byte = 0; byte < bytes;byte++)
//                     res_row[byte] = src_row[byte];
//             }
//         }
//     }
// }

template<typename T>
void launch_move_one_col(const T* __restrict__ src, T* __restrict__  res, 
                        const char* __restrict__ src_isnull, char* __restrict__ res_isnull,
                        uint32_t* __restrict__ map, // map[oldIdx] = newIdx
                        const uint32_t srcSz, cudaStream_t stream) {

    move_one_col<T><<<CEIL_DIVI(srcSz, BLOCK_DIM), BLOCK_DIM, 0, stream>>>(src, res, src_isnull, res_isnull, map, srcSz);

}


template<typename T>
void launch_move_rows_filter_kernel(T* __restrict__ src, T* res, uint32_t* __restrict__ map, bool* __restrict__ mask, char* __restrict__ isnull, const uint32_t srcSz, cudaStream_t& stream) {

    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(srcSz, threads.x * COARSENING_FACTOR));

    move_rows_filter_kernel<T><<<blocks, threads, 0, stream>>>(src, res, map, mask, isnull, srcSz);

    CUDA_CHECK_LAST();
}


template<typename T>
void lanch_move_rows_join_kernel(const T* __restrict__ src, T* __restrict__ res,
                                    const uint32_t* __restrict__ pairs, // map[oldIdx] = newIdx + 1
                                    const char* __restrict__ isnull,
                                    char* __restrict__ out_isnull,
                                    const uint32_t batchSize,
                                    const bool isright, cudaStream_t stream)  {
    dim3 threads(BLOCK_DIM);
    dim3 blocks (CEIL_DIV(batchSize+1, threads.x * COARSENING_FACTOR));
    move_rows_join_kernel<<<blocks, threads, 0, stream>>>(src, res, pairs, isnull, out_isnull, batchSize, isright);
}

    // Explicit template instantiations
    template void launch_move_rows_filter_kernel<int>(int* __restrict__, int*, uint32_t* __restrict__, bool* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<float>(float* __restrict__, float*, uint32_t* __restrict__, bool* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<int64_t>(int64_t* __restrict__, int64_t*, uint32_t* __restrict__, bool* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&);
    template void launch_move_rows_filter_kernel<String>(String* __restrict__, String*, uint32_t* __restrict__, bool* __restrict__, char* __restrict__, const uint32_t, cudaStream_t&);
    
    template void launch_move_one_col<String>(const String* __restrict__ src, String* __restrict__  res,  const char* __restrict__ src_isnull, char* __restrict__ res_isnull, uint32_t* __restrict__ map, const uint32_t srcSz, cudaStream_t stream);
    template void launch_move_one_col<int64_t>(const int64_t* __restrict__ src, int64_t* __restrict__  res,  const char* __restrict__ src_isnull, char* __restrict__ res_isnull, uint32_t* __restrict__ map, const uint32_t srcSz, cudaStream_t stream);
    template void launch_move_one_col<float>(const float* __restrict__ src, float* __restrict__  res,  const char* __restrict__ src_isnull, char* __restrict__ res_isnull, uint32_t* __restrict__ map, const uint32_t srcSz, cudaStream_t stream);
    template void launch_move_one_col<int>(const int* __restrict__ src, int* __restrict__  res,  const char* __restrict__ src_isnull, char* __restrict__ res_isnull, uint32_t* __restrict__ map, const uint32_t srcSz, cudaStream_t stream);
    // for nullset
    // template void launch_move_rows_filter_kernel<char>(char* __restrict__, char*, uint32_t* __restrict__, bool* __restrict__, bool* __restrict__, const uint32_t, cudaStream_t&);
    template void lanch_move_rows_join_kernel<int>(const int* __restrict__, int* __restrict__, const uint32_t* __restrict__, const char* __restrict__, char* __restrict__, const uint32_t, const bool, cudaStream_t) ;
    template void lanch_move_rows_join_kernel<float>(const float* __restrict__, float* __restrict__, const uint32_t* __restrict__, const char* __restrict__, char* __restrict__, const uint32_t, const bool, cudaStream_t) ;
    template void lanch_move_rows_join_kernel<int64_t>(const int64_t* __restrict__, int64_t* __restrict__, const uint32_t* __restrict__, const char* __restrict__, char* __restrict__, const uint32_t, const bool, cudaStream_t) ;
    template void lanch_move_rows_join_kernel<String>(const String* __restrict__, String* __restrict__, const uint32_t* __restrict__, const char* __restrict__, char* __restrict__, const uint32_t, const bool, cudaStream_t) ;
}
