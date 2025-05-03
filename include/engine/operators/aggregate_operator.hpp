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
    class AggregateOperator final: public Operator {
    private:
        std::vector<std::unique_ptr<our::AggregateExpression>> expressions;
    
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
        
        std::vector<cudaStream_t> streams;
        std::vector<void*> resultData(columns.size());

        std::vector<std::shared_ptr<NullBitSet>> nullset(columns.size()); // no nulls here :love
        char nonull = 0;
        // pass batch to references & store them
        uint32_t batchNum = 0;
        auto childBatchId = children[0]->next(cacheManager);
        while(childBatchId != 0) {
            // get child batch
            auto childBatch = cacheManager.getBatch(childBatchId);
            size_t batchSize = childBatch->batchSize;
            if(batchSize == 0) { // don't don anything
                continue;
            }
            streams.push_back(childBatch->stream);
            childBatch->moveTo(Device::GPU);
            // run evaluation of all
            our::ExpressionArg arg {}; 
            arg.batchs.push_back(childBatch.get());
            for(auto& expr: expressions) {
                expr->evaluate(arg);
            }

            batchNum++; //TODO: handle if empty childrens
            childBatchId = children[0]->next(cacheManager);
        }
        isFinished = true;
        // handle if empty result //TODO: should return empty result not zero since filters/filtered data
        if(batchNum == 0) 
            return 0;

        for(auto& stream: streams) {
            CUDA_CHECK( cudaStreamSynchronize(stream) );
            CUDA_CHECK( cudaStreamDestroy(stream) );
        }
        uint32_t expIdx = 0;
        for(auto& expr: expressions) {
            nullset[expIdx] = std::shared_ptr<NullBitSet>(new NullBitSet(&nonull, 1));
            resultData[expIdx++] = expr->getAggregate();
        }

        //TODO: optimize it :xd by using default but don't destroy it :xdxdxd
        auto batch = std::unique_ptr<Batch>(new Batch( resultData, Device::GPU,  1, columns, nullset));

        return cacheManager.putBatch(std::move(batch));
    }
  
private:
    // should be called only from init
    void initColumns() {
        //! should through error later
        if(children.empty()) {
            LOG_ERROR(logger, "Aggregate Operator has no children");
            std::cout << "Aggregate Operator has no children\n";
            return;
        }
        // get logical operator
        const auto& castOp = logicalOp.Cast<duckdb::LogicalAggregate>();
        
        uint32_t index = 0;
        for (auto& expr : castOp.expressions) {
            if(expr->type != duckdb::ExpressionType::AGGREGATE && expr->type != duckdb::ExpressionType::BOUND_AGGREGATE )
                throw std::runtime_error("Found non aggregate expression in aggeregate operator");

            auto our_expr = std::unique_ptr<our::AggregateExpression>(new our::AggregateExpression(*expr));

            columns.push_back(our_expr->column);
            expressions.push_back(std::move(our_expr));

        }
    }
};
}    