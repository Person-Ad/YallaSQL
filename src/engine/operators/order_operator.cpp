#include "engine/operators/list.hpp"
#include "utils/macros.hpp"
#include "cuda_runtime.h"
#include "kernels/merge_batchs.hpp"

namespace YallaSQL {

void OrderOperator::load_in_buffer(int* buffer, int &buffer_offset, int &run_offset, RUN* run, CacheManager& cacheManager, cudaStream_t stream) {
    while(buffer_offset < BUFFER_SZ && run_offset < run->total_rows) {
        auto it = std::upper_bound(run->prefix_sum_rows.begin(), run->prefix_sum_rows.end(), run_offset);
        int batch_idx = std::distance(run->prefix_sum_rows.begin(), it) - 1;
        int in_batch_idx = run_offset - run->prefix_sum_rows[batch_idx];

        auto batch = cacheManager.getBatch(run->batchs[batch_idx]);
        batch->moveTo(Device::GPU);

        int m = std::min((uint32_t)batch->batchSize - in_batch_idx,   BUFFER_SZ - buffer_offset);
        CUDA_CHECK( cudaMemcpyAsync(buffer + buffer_offset, 
                                    static_cast<char*>(batch->columnData[keyIndex]) + in_batch_idx * sizeof(int), 
                                    m * sizeof(int), 
                                    cudaMemcpyDeviceToDevice, 
                                    stream) );

        if(m < batch->batchSize - in_batch_idx) {
            BatchID newCacheIndex = cacheManager.putBatch(std::move(batch));
            run->batchs[batch_idx] = newCacheIndex;
        }
        buffer_offset += m; run_offset += m;
    }
}

void OrderOperator::shift_buffer(int *buffer, int i_last, int m, int &buffer_offset, cudaStream_t stream) {
    int shift_m = m - i_last;
    if (shift_m > 0) {
        CUDA_CHECK(cudaMemcpyAsync(buffer, 
                                    buffer + i_last, 
                                    shift_m * sizeof(int), 
                                    cudaMemcpyDeviceToDevice, 
                                    stream));
    }
    buffer_offset = shift_m;
}

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
    int* l_buffer, *r_buffer, *o_buffer;
    int *i_last_d;
    CUDA_CHECK( cudaMallocAsync((void**)&l_buffer, sizeof(int) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&r_buffer, sizeof(int) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&o_buffer, sizeof(int) * BUFFER_SZ, stream) );
    CUDA_CHECK( cudaMallocAsync((void**)&i_last_d, sizeof(int), stream) );

    // cudaDeviceSynchronize();
    // CUDA_CHECK_LAST();

    int i_last, j_last;
    int l_buffer_off = 0, r_buffer_off = 0;
    int l_run_off = 0, r_run_off = 0;
    while((l_run_off < left->total_rows && r_run_off < right->total_rows) || l_buffer_off > 0 || r_buffer_off > 0 ) {
        load_in_buffer(l_buffer, l_buffer_off, l_run_off, left, cacheManager, stream);
        load_in_buffer(r_buffer, r_buffer_off, r_run_off, right, cacheManager, stream);

        int m = l_buffer_off, n = r_buffer_off;
        int k = std::min(n + m, (int)BUFFER_SZ);
        // 
        YallaSQL::Kernel::launch_merge_sorted_array_kernel(l_buffer, r_buffer, o_buffer, i_last_d, k, m, n, stream);
        //! temprory
        std::vector<void*> newData(1);
        std::vector<std::shared_ptr<NullBitSet>> nullset(1);
        
        std::vector<std::shared_ptr<Column>> newCols(1);
        newCols[0] = std::shared_ptr<Column>(new Column("idx", DataType::INT));
        
        CUDA_CHECK( cudaMallocAsync(&newData[0], k * sizeof(int), stream) );
        CUDA_CHECK( cudaMemcpyAsync(newData[0], o_buffer, k * sizeof(int), cudaMemcpyDeviceToDevice, stream) );

        nullset[0] =  std::shared_ptr<NullBitSet>(new NullBitSet(k, stream));
        CUDA_CHECK( cudaMemsetAsync(nullset[0]->bitset, 0, k, stream) );

        auto outbatch = std::unique_ptr<Batch>(new Batch( newData, Device::GPU,  k, newCols, nullset, stream));
        
        BatchID newCacheIdx = cacheManager.putBatch(std::move(outbatch));
        // add it to run normally
        out_run->batchs.push_back(newCacheIdx);
        out_run->total_rows += k;
        out_run->prefix_sum_rows.push_back(k);

        CUDA_CHECK( cudaMemcpyAsync(&i_last, i_last_d, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        j_last = k - i_last;
        
        shift_buffer(l_buffer, i_last, m, l_buffer_off, stream);
        shift_buffer(r_buffer, j_last, n, r_buffer_off, stream);
    }

    CUDA_CHECK(cudaFreeAsync(l_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(r_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(o_buffer, stream));
    CUDA_CHECK(cudaFreeAsync(i_last_d, stream));
    delete left; delete right;
    return out_run;
}




}