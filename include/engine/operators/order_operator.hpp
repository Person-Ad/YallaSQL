// [x] test cache manager
// [ ] get how to handle ASC, DESC
// [x] sort one batch: int -> datetime -> float -> 
//          !string
// [x] wrap it in runs 
// [ ] build merging templates
// [ ] merging runs & output batchs :ISA
#pragma once
#include <unistd.h>
#include "engine/operators/expressions/list.hpp"
#include "engine/operators/operator.hpp"
#include "kernels/prefix_sum.hpp"
#include "kernels/move_rows_kernel.hpp"
#include "kernels/string_kernel.hpp"
#include "kernels/radix_sort_kernel.hpp"
#include "kernels/merge_batchs.hpp"

#include <duckdb/planner/operator/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include <memory>


struct RUN {
    int total_rows = 0;
    std::vector<BatchID> batchs;
    std::vector<int> prefix_sum_rows;
    cudaStream_t stream = nullptr;
    RUN() { prefix_sum_rows.push_back(0); }
    void init() {
        for(int i = 1; i < prefix_sum_rows.size();i++) 
            prefix_sum_rows[i] += prefix_sum_rows[i - 1];
    }

};

namespace YallaSQL {
class OrderOperator final: public Operator {
    private:
    std::vector<std::unique_ptr<our::Expression>> expressions;
    // store
    std::stack<BatchID> stack;
    bool first_pass = false;
    // sorting
    std::unique_ptr<our::BoundRefSortExpression> expression;
    duckdb::OrderType order;
    std::vector<RUN*> runs;
    uint32_t keyIndex;
    //TODO: set it
    uint32_t MAX_BATCH_SZ;
    uint32_t BUFFER_SZ;
    uint32_t currBatchIdx = 0;
    DataType keytype;

public:
    // inherit from operator
    using Operator::Operator; 
    
    void init() override  {
        if(isInitalized) return;
        // get columns & filters from logical operator
        initColumns();
        // update state
        isInitalized = true;
    }

    BatchID next(CacheManager& cacheManager) override {
        if(!isInitalized) init();
        if(isFinished || children.empty()) {isFinished = true; return 0;}

        if(!first_pass) {
            firstPass(cacheManager);
            //! test merging
            merge_all_sorted(runs, cacheManager);
            //! delete to not use it by mistake
            cudaStreamSynchronize(runs[0]->stream);
        } 
        
        if(currBatchIdx < runs[0]->batchs.size()) {
            auto batch = cacheManager.getBatch(runs[0]->batchs[currBatchIdx++]);
            cudaStreamCreate(&batch->stream);
            return cacheManager.putBatch(std::move(batch));
        }
        
        
        return 0;
    }
  
private:
    void firstPass(CacheManager& cacheManager) {
        if(first_pass) return;
        
        // 1. foreach batch in R
        BatchID batch_idx = children[0]->next(cacheManager);
        MAX_BATCH_SZ = batch_idx != 0 ? cacheManager.refBatch(batch_idx).batchSize : 0;
        columns = batch_idx != 0 ? cacheManager.refBatch(batch_idx).columns : std::vector<std::shared_ptr<Column>>();
        BUFFER_SZ = MAX_BATCH_SZ;

        while(batch_idx) {
            // 2.1 convert it to comparable format
            auto batch = cacheManager.getBatch(batch_idx);
            auto countable_batch = expression->evaluate(*batch);
            uint32_t batchSize = countable_batch.batchSize;
            // 2.2 sort batch locally
            uint32_t* d_new_index = sortLocallyBatch(countable_batch.result, batchSize, batch->stream);
            // 2.3 use d_new_index to move rows locally
            auto new_batch = YallaSQL::Kernel::move_rows_batch(*batch, d_new_index);
            keytype = new_batch->columns[keyIndex]->type;
            // 2.4 cache it no longer needed now
            BatchID referenceBatchID =  cacheManager.putBatch(std::move(new_batch));
            // 2.5 create run with the batch
            RUN* run = new RUN(); 
            run->total_rows = batchSize;
            run->batchs.push_back(referenceBatchID);
            run->prefix_sum_rows.push_back(batchSize);
            runs.push_back(run);
            // 3. get next batch
            batch_idx = children[0]->next(cacheManager);
        }

        first_pass = true;
    }
    
    void add_to_bufferBatch(Batch& src, Batch& res, int in_offset, int out_offset, int size, cudaStream_t stream);
    template <typename T>
    BatchID load_in_buffer(T* buffer, BatchID prev_buffer, int &buffer_offset, int &run_offset, RUN* run, CacheManager& cacheManager, cudaStream_t stream);
    
    template <typename T>
    BatchID shift_buffer(T *buffer, std::unique_ptr<Batch> prev_batch, int i_last, int m, int &buffer_offset, CacheManager&cacheManager, cudaStream_t stream);

    template <typename T>
    RUN* merge_two_runs(RUN* left, RUN* right, CacheManager& cacheManager) ;

    template <typename T>
    void launch_kernel_by_type(T* l_buffer, T* r_buffer, T*o_buffer, uint32_t* new_idxs, bool* table_idxs, int *i_last_d, const uint32_t k, uint32_t m, uint32_t n, cudaStream_t stream ) {
        YallaSQL::Kernel::launch_merge_sorted_array_kernel(l_buffer, r_buffer, o_buffer, new_idxs, table_idxs,  i_last_d, k, m, n, stream);
    }

    void launch_kernel_by_type(YallaSQL::Kernel::String* l_buffer, YallaSQL::Kernel::String* r_buffer, YallaSQL::Kernel::String*o_buffer, uint32_t* new_idxs, bool* table_idxs, int *i_last_d, const uint32_t k, uint32_t m, uint32_t n, cudaStream_t stream ) {
        YallaSQL::Kernel::launch_merge_sorted_array_kernel_str(l_buffer, r_buffer, o_buffer, new_idxs, table_idxs, i_last_d, k, m, n, stream);
    }

    void merge_all_sorted(std::vector<RUN*>& in_runs, CacheManager& cacheManager) { 
        std::vector<RUN*> out_runs;
        while(in_runs.size() > 1) {
            for(int i = 0; i < in_runs.size(); i+=2) {
                // add last run as no one merged with
                if(i == in_runs.size() - 1) {
                    out_runs.push_back(in_runs[in_runs.size() - 1]);
                } else {
                    RUN* new_run;
                    switch (keytype)
                    {
                    case DataType::INT:
                        new_run = merge_two_runs<int>(in_runs[i], in_runs[i+1], cacheManager);
                        break;
                    case DataType::FLOAT:
                        new_run = merge_two_runs<float>(in_runs[i], in_runs[i+1], cacheManager);
                        break;
                    case DataType::DATETIME:
                        new_run = merge_two_runs<int64_t>(in_runs[i], in_runs[i+1], cacheManager);
                        break;
                    case DataType::STRING:
                        new_run = merge_two_runs<YallaSQL::Kernel::String>(in_runs[i], in_runs[i+1], cacheManager);
                        break;
                    default:
                        break;
                    }
                    out_runs.push_back(new_run);
                }
            }
    
            in_runs = out_runs;
            out_runs = std::vector<RUN*>();
        }
    }

    // sort batchs locally and get new idxs
    [[nodiscard]] uint32_t* sortLocallyBatch(void* data, uint32_t batchSize, cudaStream_t stream) {
        uint32_t* d_new_index;
        switch (order) {
        case duckdb::OrderType::ORDER_DEFAULT:
        case duckdb::OrderType::ASCENDING:
            switch (expression->returnType) {
            case DataType::FLOAT:
            case DataType::INT:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::AscOp<uint32_t>, uint32_t, BLOCK_DIM>(
                    static_cast<uint32_t*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::AscOp<uint64_t>, uint64_t, BLOCK_DIM>(
                    static_cast<uint64_t*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            case DataType::STRING:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::AscOpString, YallaSQL::Kernel::String, BLOCK_DIM_STR>(
                    static_cast<YallaSQL::Kernel::String*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            }
            break;
        
        case duckdb::OrderType::DESCENDING:
            switch (expression->returnType) {
            case DataType::FLOAT:
            case DataType::INT:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::DescOp<uint32_t>, uint32_t, BLOCK_DIM>(
                    static_cast<uint32_t*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::DescOp<uint64_t>, uint64_t, BLOCK_DIM>(
                    static_cast<uint64_t*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            case DataType::STRING:
                YallaSQL::Kernel::launch_radix_sort<YallaSQL::Kernel::DescOpString, YallaSQL::Kernel::String, BLOCK_DIM_STR>(
                    static_cast<YallaSQL::Kernel::String*>(data), 
                    d_new_index, 
                    batchSize,
                    stream
                );
                break;
            }
            break;
        
        }
        
        
        
        return d_new_index;
    }
    // should be called only from init
    void initColumns() {
        //! should through error later
        if(children.empty()) {
            LOG_ERROR(logger, "Projection Operator has no children");
            std::cout << "Projection Operator has no children\n";
            return;
        }
        // get logical operator
        const auto& castOp = logicalOp.Cast<duckdb::LogicalOrder>();
        
        const auto& order_ = castOp.orders[0];
        order = order_.type;
        expression = std::unique_ptr<our::BoundRefSortExpression> (new our::BoundRefSortExpression(*order_.expression));
        keyIndex = expression->idx;
        printf("======= %i ==========\n", keyIndex);
    }
};
}    