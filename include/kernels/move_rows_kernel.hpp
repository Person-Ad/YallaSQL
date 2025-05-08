#pragma once
#include "kernels/string_kernel.hpp"
#include "utils/macros.hpp"
#include "kernels/constants.hpp"
#include "batch.hpp"
#include <cuda_runtime.h>
#include "null_bit_set.hpp"

namespace YallaSQL::Kernel {

    template<typename T>
    __global__ void move_rows_filter_kernel(T* __restrict__ src, T* res,
                                        uint32_t* __restrict__ map, // map[oldIdx] = newIdx + 1
                                        bool* __restrict__ mask,
                                        char* __restrict__ isnull,
                                        const uint32_t srcSz) ;

    template<typename T>
    void launch_move_rows_filter_kernel(T* __restrict__ src, T* res, 
                                        uint32_t* __restrict__ map, 
                                        bool* __restrict__ mask, 
                                        char* __restrict__ isnull,
                                        const uint32_t srcSz, 
                                        cudaStream_t& stream);                                        

    template<typename T>
    void launch_move_one_col(const T* __restrict__ src, T* __restrict__  res, 
                            const char* __restrict__ src_isnull, char* __restrict__ res_isnull,
                            uint32_t* __restrict__ map, // map[oldIdx] = newIdx
                            const uint32_t srcSz, cudaStream_t stream);
                 
    template<typename T>
    void lanch_move_rows_join_kernel(const T* __restrict__ src, T* __restrict__ res,
        const uint32_t* __restrict__ pairs, // map[oldIdx] = newIdx + 1
        const char* __restrict__ isnull,
        char* __restrict__ out_isnull,
        const uint32_t batchSize,
        const bool isright, cudaStream_t stream);


        
    inline std::unique_ptr<Batch> move_rows_batch(Batch& src_batch, uint32_t* __restrict__ map) {
        std::vector<void*> newColumnData(src_batch.columns.size()); 
        std::vector<std::shared_ptr<NullBitSet>> nullset(src_batch.columns.size()); 
        int newBatchSize = src_batch.batchSize;
        cudaStream_t stream = src_batch.stream;

        int currIdx = 0;
        for(auto& oldCol: src_batch.columnData) {
            CUDA_CHECK( cudaMallocAsync(&newColumnData[currIdx], newBatchSize * src_batch.columns[currIdx]->bytes, stream)  );
            nullset[currIdx] = std::shared_ptr<NullBitSet>(new NullBitSet(newBatchSize, stream));
            switch (src_batch.columns[currIdx]->type){
                case DataType::INT:
                    launch_move_one_col(static_cast<int*>(oldCol), static_cast<int*>(newColumnData[currIdx]), src_batch.nullset[currIdx]->bitset, nullset[currIdx]->bitset,  map, newBatchSize, stream);
                break;
                case DataType::FLOAT:
                    launch_move_one_col(static_cast<float*>(oldCol), static_cast<float*>(newColumnData[currIdx]), src_batch.nullset[currIdx]->bitset, nullset[currIdx]->bitset,  map, newBatchSize, stream);
                    break;
                case DataType::DATETIME:
                    launch_move_one_col(static_cast<int64_t*>(oldCol), static_cast<int64_t*>(newColumnData[currIdx]), src_batch.nullset[currIdx]->bitset, nullset[currIdx]->bitset,  map, newBatchSize, stream);
                    break;
                case DataType::STRING:
                    launch_move_one_col(static_cast<String*>(oldCol), static_cast<String*>(newColumnData[currIdx]), src_batch.nullset[currIdx]->bitset, nullset[currIdx]->bitset,  map, newBatchSize, stream);
                    break;
                default:
                    break;
                }

            currIdx++;
        }

        auto batch = std::unique_ptr<Batch>(new Batch( newColumnData, Device::GPU,  newBatchSize, src_batch.columns, nullset, stream));
        return batch;
    }
    // Batch* launch_move_rows(Batch& src_batch) {
    //     src_batch.moveTo(Device::GPU);
    //     int ncols = src_batch.columns.size();
    //     int nrows = src_batch.batchSize;

    //     auto dist_batch = new SimplifiedBatch();
    //     dist_batch->ncols = ncols;
    //     dist_batch->nrows = nrows;

    //     // Allocate device memory for columns array
    //     CUDA_CHECK(cudaMalloc(&dist_batch->columns, sizeof(void*) * ncols));
    //     CUDA_CHECK(cudaMalloc(&dist_batch->types, sizeof(DataType) * ncols));
        
    //     // Copy column pointers
    //     CUDA_CHECK(cudaMemcpy(dist_batch->columns, 
    //                         src_batch.columnData.data(), 
    //                         sizeof(void*) * ncols, 
    //                         cudaMemcpyHostToDevice));
    //     // Copy types
        

    //     // Copy column types
    //     for (int i = 0; i < ncols; ++i) {
    //         DataType type = src_batch.columns[i]->type;
    //         CUDA_CHECK(cudaMemcpy(dist_batch->types + i,  &type, sizeof(DataType), cudaMemcpyHostToDevice));
    //     }

    //     return dist_batch;
    // }

}