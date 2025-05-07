#include "engine/operators/list.hpp"
#include "utils/macros.hpp"
#include "cuda_runtime.h"
#include "kernels/merge_batchs.hpp"

namespace YallaSQL {

void OrderOperator::load_in_buffer(int* buffer, int &buffer_offset, int &run_offset, RUN* run, CacheManager& cacheManager) {
    while(buffer_offset < BUFFER_SZ && run_offset < run->total_rows) {
        auto it = std::upper_bound(run->prefix_sum_rows.begin(), run->prefix_sum_rows.end(), run_offset);
        int batch_idx = std::distance(run->prefix_sum_rows.begin(), it) - 1;
        int in_batch_idx = run_offset - run->prefix_sum_rows[batch_idx];

        auto batch = cacheManager.getBatch(run->batchs[batch_idx]);
        batch->moveTo(Device::GPU);
        cudaStreamSynchronize(batch->stream);
        CUDA_CHECK_LAST();
        int m = std::min((uint32_t)batch->batchSize - in_batch_idx,   BUFFER_SZ - buffer_offset);
        CUDA_CHECK( cudaMemcpy(buffer + buffer_offset, static_cast<char*>(batch->columnData[keyIndex]) + in_batch_idx, m * sizeof(int), cudaMemcpyDeviceToDevice) );

        if(m < batch->batchSize - in_batch_idx) {
            BatchID newCacheIndex = cacheManager.putBatch(std::move(batch));
            run->batchs[batch_idx] = newCacheIndex;
        }
        buffer_offset += m; run_offset += m;
    }
}

void OrderOperator::shift_buffer(int *buffer, int i_last, int m, int &buffer_offset) {
    int shift_m = m - i_last;
    if (shift_m > 0) {
        CUDA_CHECK(cudaMemcpy(buffer, buffer + i_last, shift_m * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    buffer_offset = shift_m;
}

RUN* OrderOperator::merge_two_runs(RUN* left, RUN* right, CacheManager& cacheManager) {
    left->init();
    right->init();
    cudaDeviceSynchronize();
    RUN* out_run = new RUN();
    int* l_buffer, *r_buffer, *o_buffer;
    int *i_last_d;
    CUDA_CHECK( cudaMalloc((void**)&l_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&r_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&o_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&i_last_d, sizeof(int)) );

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    int i_last, j_last;
    int l_buffer_off = 0, r_buffer_off = 0;
    int l_run_off = 0, r_run_off = 0;
    while((l_run_off < left->total_rows && r_run_off < right->total_rows) || l_buffer_off > 0 || r_buffer_off > 0 ) {
        load_in_buffer(l_buffer, l_buffer_off, l_run_off, left, cacheManager);
        load_in_buffer(r_buffer, r_buffer_off, r_run_off, right, cacheManager);
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();

        int m = l_buffer_off, n = r_buffer_off;
        int k = std::min(n + m, (int)BUFFER_SZ);
        // 
        YallaSQL::Kernel::launch_merge_sorted_array_kernel(l_buffer, r_buffer, o_buffer, i_last_d, k, m, n);
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();
        //! temprory
        std::vector<void*> newData(1);
        std::vector<std::shared_ptr<NullBitSet>> nullset(1);
        std::vector<std::shared_ptr<Column>> newCols(1);
        newCols[0] = std::shared_ptr<Column>(new Column("idx", DataType::INT));
        cudaMallocHost(&newData[0], k * sizeof(int));
        CUDA_CHECK( cudaMemcpy(newData[0], o_buffer, k * sizeof(int), cudaMemcpyDeviceToHost) );
        nullset[0] =  std::shared_ptr<NullBitSet>(new NullBitSet(k, cudaStreamDefault));
        CUDA_CHECK( cudaMemset(nullset[0]->bitset, 0, k) );
        nullset[0]->moveToCpu(k, cudaStreamDefault);
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();
        auto outbatch = std::unique_ptr<Batch>(new Batch( newData, Device::CPU,  k, newCols, nullset));
        BatchID newCacheIdx = cacheManager.putBatch(std::move(outbatch));
        // add it to run normally
        out_run->batchs.push_back(newCacheIdx);
        out_run->total_rows += k;
        out_run->prefix_sum_rows.push_back(k);

        cudaDeviceSynchronize();
        CUDA_CHECK_LAST();
        CUDA_CHECK( cudaMemcpy(&i_last, i_last_d, sizeof(int), cudaMemcpyDeviceToHost) );
        j_last = k - i_last;
        
        shift_buffer(l_buffer, i_last, m, l_buffer_off);
        shift_buffer(r_buffer, j_last, n, r_buffer_off);
    }

    CUDA_CHECK(cudaFree(l_buffer));
    CUDA_CHECK(cudaFree(r_buffer));
    CUDA_CHECK(cudaFree(o_buffer));
    CUDA_CHECK(cudaFree(i_last_d));
    delete left; delete right;
    return out_run;
}




}