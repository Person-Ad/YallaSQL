// [x] test cache manager
// [ ] get how to handle ASC, DESC
// [x] sort one batch: int -> datetime -> float -> 
//          !string
// [ ] wrap it in runs 
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

#include <duckdb/planner/operator/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include <memory>


namespace YallaSQL::KernelTest {
        void launch_radix_sort(uint32_t* d_input, uint32_t* &d_idxs_in, const uint32_t N);
}
struct RUN {
    int total_rows = 0;
    std::vector<BatchID> batchs;
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

        //!test cache manager
        if(!first_pass) 
            firstPass(cacheManager);
        if(!stack.empty()) {
            BatchID top_batch = stack.top();
            stack.pop();

            // auto batch = cacheManager.getBatch(top_batch);

            return top_batch;
        }
        
        
        return 0;
    }
  
private:
    void firstPass(CacheManager& cacheManager) {
        if(first_pass) return;
        
        // 1. foreach batch in R
        BatchID batch_idx = children[0]->next(cacheManager);
        while(batch_idx) {
            // 2.1 convert it to comparable format
            auto batch = cacheManager.getBatch(batch_idx);
            auto countable_batch = expression->evaluate(*batch);
            // 2.2 sort batch locally
            uint32_t* d_new_index = sortLocallyBatch(countable_batch.result, countable_batch.batchSize, batch->stream);
            // 2.3 use d_new_index to move rows locally
            auto new_batch = YallaSQL::Kernel::move_rows_batch(*batch, d_new_index);
            // 2.4 cache it no longer needed now
            BatchID referenceBatchID =  cacheManager.putBatch(std::move(new_batch));
            stack.push(referenceBatchID);
            // 3. get next batch
            batch_idx = children[0]->next(cacheManager);
        }

        first_pass = true;
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

    }
};
}    