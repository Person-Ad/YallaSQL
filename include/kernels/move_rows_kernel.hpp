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

    template<typename T>
    void launch_move_one_col_merge(const T* __restrict__ Asrc, const T* __restrict__ Bsrc, 
                            T* __restrict__  res,
                            const char* __restrict__ Aisnull, const char* __restrict__ Bisnull,
                            char* __restrict__ res_isnull,
                            const uint32_t* __restrict__ map, // map[newIdx] = oldIdx
                            const bool* __restrict__ tableIdxs,
                            bool isAsc,
                            const uint32_t srcSz, cudaStream_t stream);

        
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

    inline std::unique_ptr<Batch> move_rows_batch_merge(Batch& Abatch, Batch& Bbatch, uint32_t* __restrict__ newIdxs, bool* __restrict__ tableIdxs, const uint32_t srcSz, cudaStream_t stream, bool isAsc = true) {
        int ncols = Abatch.columns.size();
        std::vector<void*> newColumnData(ncols); 
        std::vector<std::shared_ptr<NullBitSet>> nullset(ncols); 
        int newBatchSize = srcSz;

        for(int currIdx = 0;currIdx < ncols;currIdx++) {
            CUDA_CHECK( cudaMallocAsync(&newColumnData[currIdx], newBatchSize * Abatch.columns[currIdx]->bytes, stream)  );
            nullset[currIdx] = std::shared_ptr<NullBitSet>(new NullBitSet(newBatchSize, stream));
            switch (Abatch.columns[currIdx]->type){
                case DataType::INT:
                    launch_move_one_col_merge(
                        static_cast<int*>(Abatch.columnData[currIdx]), 
                        static_cast<int*>(Bbatch.columnData[currIdx]), 
                        static_cast<int*>(newColumnData[currIdx]), 
                        Abatch.nullset[currIdx]->bitset, 
                        Bbatch.nullset[currIdx]->bitset, 
                        nullset[currIdx]->bitset,  
                        newIdxs, tableIdxs, isAsc, newBatchSize, stream);
                break;
                case DataType::FLOAT:
                    launch_move_one_col_merge(
                        static_cast<float*>(Abatch.columnData[currIdx]), 
                        static_cast<float*>(Bbatch.columnData[currIdx]), 
                        static_cast<float*>(newColumnData[currIdx]), 
                        Abatch.nullset[currIdx]->bitset, 
                        Bbatch.nullset[currIdx]->bitset, 
                        nullset[currIdx]->bitset,  
                        newIdxs, tableIdxs, isAsc, newBatchSize, stream);
                    break;
                case DataType::DATETIME:
                    launch_move_one_col_merge(
                        static_cast<int64_t*>(Abatch.columnData[currIdx]), 
                        static_cast<int64_t*>(Bbatch.columnData[currIdx]), 
                        static_cast<int64_t*>(newColumnData[currIdx]), 
                        Abatch.nullset[currIdx]->bitset, 
                        Bbatch.nullset[currIdx]->bitset, 
                        nullset[currIdx]->bitset,  
                        newIdxs, tableIdxs, isAsc, newBatchSize, stream);
                    break;
                case DataType::STRING:
                    launch_move_one_col_merge(
                        static_cast<String*>(Abatch.columnData[currIdx]), 
                        static_cast<String*>(Bbatch.columnData[currIdx]), 
                        static_cast<String*>(newColumnData[currIdx]), 
                        Abatch.nullset[currIdx]->bitset, 
                        Bbatch.nullset[currIdx]->bitset, 
                        nullset[currIdx]->bitset,  
                        newIdxs, tableIdxs, isAsc, newBatchSize, stream);
                    break;
                default:
                    break;
                }

        }

        auto batch = std::unique_ptr<Batch>(new Batch( newColumnData, Device::GPU,  newBatchSize, Abatch.columns, nullset, stream));
        return batch;
    }




}