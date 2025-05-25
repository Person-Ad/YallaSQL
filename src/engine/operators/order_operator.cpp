#include "engine/operators/list.hpp"
#include "utils/macros.hpp"
#include "cuda_runtime.h"
#include "kernels/merge_batchs.hpp"

namespace YallaSQL {

template RUN* OrderOperator::merge_two_runs<int>(RUN*, RUN*, CacheManager&);
template RUN* OrderOperator::merge_two_runs<float>(RUN*, RUN*, CacheManager&);
template RUN* OrderOperator::merge_two_runs<int64_t>(RUN*, RUN*, CacheManager&);
template RUN* OrderOperator::merge_two_runs<YallaSQL::Kernel::String>(RUN*, RUN*, CacheManager&);

template <typename T>
BatchID OrderOperator::load_in_buffer(T* buffer, BatchID prev_buffer, int &buffer_offset, int &run_offset, RUN* run, CacheManager& cacheManager, cudaStream_t stream) {
    std::unique_ptr<Batch> result_batch;
    if(prev_buffer == 0) {
        std::vector<void*> newColumnData(columns.size()); 
        std::vector<std::shared_ptr<NullBitSet>> nullset(columns.size()); 
    
        // Allocate memory for each column and nullset in the result batch
        for (size_t i = 0; i < columns.size(); ++i) {
            CUDA_CHECK(cudaMallocAsync(&newColumnData[i], BUFFER_SZ * columns[i]->bytes, stream));
            nullset[i] = std::make_shared<NullBitSet>(BUFFER_SZ, stream);
        }
        // Create a new batch for storing aggregated data

        result_batch = std::make_unique<Batch>(newColumnData, Device::GPU,  BUFFER_SZ, columns, nullset, stream);
    } else {
        result_batch = cacheManager.getBatch(prev_buffer);
        result_batch->moveTo(Device::GPU);
    }
    

    while(buffer_offset < BUFFER_SZ && run_offset < run->total_rows) {
        auto it = std::upper_bound(run->prefix_sum_rows.begin(), run->prefix_sum_rows.end(), run_offset);
        int batch_idx = std::distance(run->prefix_sum_rows.begin(), it) - 1;
        int in_batch_idx = run_offset - run->prefix_sum_rows[batch_idx];

        auto batch = cacheManager.getBatch(run->batchs[batch_idx]);
        batch->moveTo(Device::GPU);

        int m = std::min((uint32_t)batch->batchSize - in_batch_idx,   BUFFER_SZ - buffer_offset);

        CUDA_CHECK( cudaMemcpyAsync(buffer + buffer_offset, 
                                    static_cast<char*>(batch->columnData[keyIndex]) + in_batch_idx * sizeof(T), 
                                    m * sizeof(T), 
                                    cudaMemcpyDeviceToDevice, 
                                    stream) );
                                    
        add_to_bufferBatch(*batch, *result_batch, in_batch_idx, buffer_offset, m, stream);
        if(m < batch->batchSize - in_batch_idx) {
            BatchID newCacheIndex = cacheManager.putBatch(std::move(batch));
            run->batchs[batch_idx] = newCacheIndex;
        }
        
        buffer_offset += m; run_offset += m;
    }
    return cacheManager.putBatch(std::move(result_batch));
}


void OrderOperator::add_to_bufferBatch(Batch& src, Batch& res, int in_offset, int out_offset, int size, cudaStream_t stream) {
    for(int i = 0;i < src.columns.size();i++) {
        CUDA_CHECK( cudaMemcpyAsync(
            static_cast<char*>(res.columnData[i]) + src.columns[i]->bytes * out_offset,
            static_cast<char*>(src.columnData[i]) + src.columns[i]->bytes * in_offset,
            src.columns[i]->bytes * size,
            cudaMemcpyDeviceToDevice,
            stream
        ));

        CUDA_CHECK( cudaMemcpyAsync(
            res.nullset[i]->bitset + out_offset,
            src.nullset[i]->bitset + in_offset,
            size,
            cudaMemcpyDeviceToDevice,
            stream
        ));
    }
}

template <typename T>
BatchID OrderOperator::shift_buffer(T *buffer, std::unique_ptr<Batch> result_batch, int i_last, int m, int &buffer_offset, CacheManager&cacheManager, cudaStream_t stream) {
    int shift_m = m - i_last;
    buffer_offset = shift_m;
    
    if (shift_m > 0) {
        CUDA_CHECK(cudaMemcpyAsync(buffer, 
                                    buffer + i_last, 
                                    shift_m * sizeof(T), 
                                    cudaMemcpyDeviceToDevice, 
                                    stream));
        // Shift all columns and nullsets in result_batch
        for (size_t i = 0; i < result_batch->columns.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(result_batch->columnData[i]),
                static_cast<char*>(result_batch->columnData[i]) + result_batch->columns[i]->bytes * i_last,
                result_batch->columns[i]->bytes * shift_m,
                cudaMemcpyDeviceToDevice,
                stream
            ));

            CUDA_CHECK(cudaMemcpyAsync(
                result_batch->nullset[i]->bitset,
                result_batch->nullset[i]->bitset + i_last,
                shift_m,
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
        BatchID newCacheIndex = cacheManager.putBatch(std::move(result_batch));
        return newCacheIndex;
    }
    return 0;
}
template<typename T>
RUN* OrderOperator::merge_two_runs(RUN* left, RUN* right, CacheManager& cacheManager) {
    left->init();
    right->init();
    CUDA_CHECK(cudaStreamSynchronize(left->stream ? left->stream : 0));
    CUDA_CHECK(cudaStreamSynchronize(right->stream ? right->stream : 0));
    
    cudaStream_t stream; 
    CUDA_CHECK(cudaStreamCreate(&stream));
    // cudaDeviceSynchronize();
    RUN* out_run = new RUN();
    out_run->stream = stream;
    int expected_total_rows = left->total_rows + right->total_rows;
    
    T* l_buffer, *r_buffer, *o_buffer;
    int *i_last_d;
    uint32_t* new_idxs; bool* table_idxs;

    CUDA_CHECK( cudaMallocAsync((void**)&l_buffer, sizeof(T) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&r_buffer, sizeof(T) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&o_buffer, sizeof(T) * BUFFER_SZ, stream) );

    CUDA_CHECK( cudaMallocAsync((void**)&new_idxs, sizeof(uint32_t) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&table_idxs, sizeof(bool) * BUFFER_SZ, stream) );

    CUDA_CHECK( cudaMallocAsync((void**)&i_last_d, sizeof(int), stream) );

    // cudaDeviceSynchronize();
    // CUDA_CHECK_LAST();

    int i_last, j_last;
    int l_buffer_off = 0, r_buffer_off = 0;
    int l_run_off = 0, r_run_off = 0;
    BatchID leftBatchID = 0, rightBatchID = 0;
    int ncols = 0;
    while((l_run_off < left->total_rows && r_run_off < right->total_rows) || l_buffer_off > 0 || r_buffer_off > 0) {
        leftBatchID = load_in_buffer(l_buffer, leftBatchID, l_buffer_off, l_run_off, left, cacheManager, stream);
        rightBatchID = load_in_buffer(r_buffer, rightBatchID, r_buffer_off, r_run_off, right, cacheManager, stream);

        int m = l_buffer_off, n = r_buffer_off;
        int k = std::min(n + m, (int)BUFFER_SZ);
        // 
        // YallaSQL::Kernel::launch_merge_sorted_array_kernel(l_buffer, r_buffer, o_buffer, i_last_d, k, m, n, stream);
        launch_kernel_by_type(l_buffer, r_buffer, o_buffer, new_idxs, table_idxs, i_last_d, k, m, n, stream);

        
        auto lbatch = cacheManager.getBatch(leftBatchID);
        auto rbatch = cacheManager.getBatch(rightBatchID);

        lbatch->moveTo(Device::GPU);
        rbatch->moveTo(Device::GPU);
        ncols = lbatch->columns.size();
        auto outBatch = YallaSQL::Kernel::move_rows_batch_merge(*lbatch, *rbatch, new_idxs, table_idxs, k, stream);
        BatchID newCacheIdx = cacheManager.putBatch(std::move(outBatch));
        // add it to run normally
        out_run->batchs.push_back(newCacheIdx);
        out_run->total_rows += k;
        out_run->prefix_sum_rows.push_back(k);

        CUDA_CHECK( cudaMemcpyAsync(&i_last, i_last_d, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        j_last = k - i_last;
        
        leftBatchID = shift_buffer(l_buffer, std::move(lbatch), i_last, m, l_buffer_off, cacheManager, stream);
        rightBatchID = shift_buffer(r_buffer, std::move(rbatch), j_last, n, r_buffer_off, cacheManager,stream);
    }
    auto emptyBatch = std::make_unique<Batch>(stream, ncols);
    // move rest of rows
    uint32_t h_new_idxs[BUFFER_SZ]; bool h_table_idx[BUFFER_SZ];
    for(int i = 0;i < BUFFER_SZ;i++) h_new_idxs[i] = i, h_table_idx[i] = true;
    CUDA_CHECK( cudaMemcpyAsync(new_idxs, h_new_idxs, BUFFER_SZ*sizeof(uint32_t), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpyAsync(table_idxs, h_table_idx, BUFFER_SZ, cudaMemcpyHostToDevice) );
    if(r_run_off < right->total_rows) {
        std::swap(l_buffer, r_buffer);
        std::swap(leftBatchID, rightBatchID);
        std::swap(l_buffer_off, r_buffer_off);
        std::swap(l_run_off, r_run_off);
        std::swap(left, right);
    }
    while(l_run_off < left->total_rows) {
        //TODO:
        leftBatchID = load_in_buffer(l_buffer, leftBatchID, l_buffer_off, l_run_off, left, cacheManager, stream);
        int m = l_buffer_off;
        int k = std::min(m, (int)BUFFER_SZ);
        int i_last = m;

        auto lbatch = cacheManager.getBatch(leftBatchID);
        lbatch->moveTo(Device::GPU);
        auto outBatch = YallaSQL::Kernel::move_rows_batch_merge(*lbatch, *emptyBatch, new_idxs, table_idxs, k, stream);
        BatchID newCacheIdx = cacheManager.putBatch(std::move(outBatch));
        // add it to run normally
        out_run->batchs.push_back(newCacheIdx);
        out_run->total_rows += k;
        out_run->prefix_sum_rows.push_back(k);

        leftBatchID = shift_buffer(l_buffer, std::move(lbatch), i_last, m, l_buffer_off, cacheManager, stream);
    }
    CUDA_CHECK(cudaFreeAsync(l_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(r_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(o_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(i_last_d, stream));
    CUDA_CHECK(cudaFreeAsync(new_idxs, stream));
    CUDA_CHECK(cudaFreeAsync(table_idxs, stream));
    delete left; delete right;
    return out_run;
}




}