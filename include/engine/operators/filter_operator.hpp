#pragma once
#include "engine/operators/expressions/list.hpp"
#include "engine/operators/operator.hpp"
#include "kernels/prefix_sum.hpp"
#include "kernels/move_rows_kernel.hpp"
#include "kernels/string_kernel.hpp"

#include <duckdb/planner/operator/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include <memory>

namespace YallaSQL {
class FilterOperator final: public Operator {
private:
    std::vector<std::unique_ptr<our::Expression>> expressions;

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
        // get batchId from children
        const auto childBatchId = children[0]->next(cacheManager);
        if(childBatchId == 0) {isFinished = true; return 0;}
        //? if no expressions | happen due to join
        if(expressions.size() == 0) {
            return childBatchId;
        }
        // get ownership of child
        auto childBatch = cacheManager.getBatch(childBatchId);
        cudaStream_t stream = childBatch->stream;
        childBatch->moveTo(Device::GPU);
        // 1. get mask
        our::ExpressionArg arg {};
        arg.batchs.push_back(childBatch.get());

        // evaluate expression
        auto exprResult = expressions[0]->evaluate(arg);
        bool *mask = static_cast<bool*>(exprResult.result);
        size_t oldBatchSize = exprResult.batchSize;

        // 2. cast to uint32 + get prefix sum
        uint32_t* map;
        CUDA_CHECK(cudaMallocAsync((void**)&map, oldBatchSize * sizeof(uint32_t), stream));
        YallaSQL::Kernel::launch_prefix_sum_mask(mask, map, oldBatchSize, stream);

        // 3. move & cpy
        uint32_t newBatchSize; 
        CUDA_CHECK( cudaMemcpyAsync(&newBatchSize, &map[oldBatchSize - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) );

        std::vector<void*> newColumnData(columns.size()); 
        std::vector<std::shared_ptr<NullBitSet>> nullset(columns.size()); 
        uint32_t currIdx = 0;
        // move col by col
        for(auto& oldCol: childBatch->columnData) {
            CUDA_CHECK( cudaMallocAsync(&newColumnData[currIdx], newBatchSize * columns[currIdx]->bytes, stream)  );
            switch (columns[currIdx]->type){
            case DataType::INT:
                YallaSQL::Kernel::launch_move_rows_filter_kernel(static_cast<int*>(oldCol), static_cast<int*>(newColumnData[currIdx]), map, mask, childBatch->nullset[currIdx]->bitset, oldBatchSize, stream);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_move_rows_filter_kernel(static_cast<float*>(oldCol), static_cast<float*>(newColumnData[currIdx]), map, mask, childBatch->nullset[currIdx]->bitset, oldBatchSize, stream);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_move_rows_filter_kernel(static_cast<int64_t*>(oldCol), static_cast<int64_t*>(newColumnData[currIdx]), map, mask, childBatch->nullset[currIdx]->bitset, oldBatchSize, stream);
                break;
            case DataType::STRING:
                YallaSQL::Kernel::launch_move_rows_filter_kernel(static_cast<YallaSQL::Kernel::String*>(oldCol), static_cast<YallaSQL::Kernel::String*>(newColumnData[currIdx]), map, mask, childBatch->nullset[currIdx]->bitset, oldBatchSize, stream);
                break;
            default:
                break;
            }
            // null set
            nullset[currIdx] = std::shared_ptr<NullBitSet>(new NullBitSet(newBatchSize, stream));
            CUDA_CHECK( cudaMemsetAsync(nullset[currIdx]->bitset, 0, newBatchSize, stream)); // set all to no nulls

            currIdx++;
        }
        // free data
        CUDA_CHECK(cudaFreeAsync(mask, stream));
        CUDA_CHECK(cudaFreeAsync(map, stream));
        // LOG_TRACE_L2(logger, "Filtered from {} to {}", oldBatchSize, newBatchSize);
        auto batch = std::unique_ptr<Batch>(new Batch( newColumnData, Device::GPU,  newBatchSize, columns, nullset, stream));

        return cacheManager.putBatch(std::move(batch));
    }


private:
    // initiate schema of operator
    // should be called only from init
    void initColumns() {
        //! should through error later
        if(children.empty()) {
            LOG_ERROR(logger, "Filter Operator has no children");
            std::cout << "Filter Operator has no children\n";
            return;
        }
        // get logical operator
        const auto& castOp = logicalOp.Cast<duckdb::LogicalFilter>();
        // create root expression
        build_expressions(castOp.expressions);
        // we just filter rows
        children[0]->init();
        columns = children[0]->columns;
    }

    // if there's multiple individual expressions in filter -> and them
    void build_expressions(const std::vector<duckdb::unique_ptr<duckdb::Expression>>& duckExpressions) {
        if(duckExpressions.size() == 0) {
            // LOG_ERROR(logger, "No expressions in filter operation");
            // std::cout << "No expressions in filter operation\n";
            // throw std::runtime_error("No expressions in filter operation");
            return;
        }

        if(duckExpressions.size() == 1) {
            auto our_expr = our::Expression::createExpression(*duckExpressions[0]);
            expressions.push_back(std::move(our_expr));
            return;
        }

        // build and tree
        auto left = our::Expression::createExpression(*duckExpressions[0]);
        auto right = our::Expression::createExpression(*duckExpressions[1]);

        left = std::unique_ptr<our::Expression>(new our::ConjuntionExpression(
            std::move(left),
            std::move(right)
        ));
        for(uint32_t i = 2; i < duckExpressions.size();i++) {
            right = our::Expression::createExpression(*duckExpressions[i]);
            left = std::unique_ptr<our::Expression>(new our::ConjuntionExpression(
                std::move(left),
                std::move(right)
            ));
        }

        expressions.push_back(std::move(left));        
        
    }
};

}