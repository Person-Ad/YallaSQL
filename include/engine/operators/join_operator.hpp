#pragma once
#include <unistd.h>
#include "engine/operators/expressions/list.hpp"
#include "engine/operators/operator.hpp"
#include "kernels/prefix_sum.hpp"
#include "kernels/move_rows_kernel.hpp"
#include "kernels/string_kernel.hpp"
#include "kernels/radix_sort_kernel.hpp"
#include "kernels/cross_product_kernel.hpp"

#include <duckdb/planner/operator/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include <memory>

namespace YallaSQL {
class JoinOperator final: public Operator {
    private:
    std::unique_ptr<our::Expression> expression;
    std::unique_ptr<Batch> rbatch = nullptr;
    std::stack<BatchID> left_stack[2];
    bool isFirstPass = false;
    bool currStack = 0;

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
        // get all left children
        if(!isFirstPass) cacheLeftChild(cacheManager);
        // get next right if I need
        if(rbatch == nullptr || left_stack[currStack].empty()) {
            if(!getRightChild(cacheManager)) {
                isFinished = true; 
                return 0;
            } 
            // recheck if not empty
            if(rbatch->batchSize == 0) { // don't don anything
                return cacheManager.putBatch(std::move(rbatch));
            }
            // recheck if should swap left stack
            if(left_stack[currStack].empty()) {
                currStack = !currStack;
            }
        }
        

        // ========= merging batch wil all left batchs ==============
        std::unique_ptr<Batch> lbatch = nullptr;
        // loop till find valid lbatchs
        while((!lbatch || lbatch->batchSize == 0) && !left_stack[currStack].empty()) {
            const auto l_batchId = left_stack[currStack].top();  
            left_stack[currStack].pop();
            lbatch = cacheManager.getBatch(l_batchId);
        } 
        
        // cross product :yalla
        lbatch->moveTo(Device::GPU);
        rbatch->moveTo(Device::GPU);
        cudaStreamSynchronize(lbatch->stream);
        cudaStreamSynchronize(rbatch->stream);
        // 1. get pairs
        cudaStream_t stream; 
        CUDA_CHECK(cudaStreamCreate(&stream));

        our::ExpressionArg arg {};
        arg.batchs.push_back(lbatch.get());
        arg.batchs.push_back(rbatch.get());
        arg.stream = stream;
        // cudaDeviceSynchronize();
        // evaluate expression
        auto exprResult = expression->evaluate(arg);
        uint32_t* pairs = static_cast<uint32_t*>(exprResult.result);
        // get batch size
        //! alwayse sync with left
        int* d_batchSize = exprResult.d_batchSize;
        int batchSize; 
        CUDA_CHECK( cudaMemcpyAsync(&batchSize, d_batchSize, sizeof(int), cudaMemcpyDeviceToHost, stream) );
        auto batch = getOutBatch(*lbatch, *rbatch, pairs, batchSize, stream);
        // free data
        CUDA_CHECK(cudaFreeAsync(pairs, stream));
        CUDA_CHECK(cudaFreeAsync(d_batchSize, stream));
        // LOG_TRACE_L2(logger, "Joined where l:{} and r:{}", lbatch->batchSize rbatch->batchSize);
        // auto batch = std::unique_ptr<Batch>(new Batch( newColumnData, Device::GPU,  newBatchSize, columns, nullset, stream));
        auto lbatchNewId = cacheManager.putBatch(std::move(lbatch));
        left_stack[!currStack].push(lbatchNewId);

        return cacheManager.putBatch(std::move(batch));

        // return 0;
    }
  
private:


std::unique_ptr<Batch> getOutBatch(Batch& l, Batch& r, uint32_t* pairs, int batchSize, cudaStream_t stream) {
        int lcols = l.columns.size(), rcols = r.columns.size();
        int ncols =  lcols+rcols ;
        std::vector<void*> resultData(ncols);
        std::vector<std::shared_ptr<NullBitSet>> nullset(ncols);
        std::vector<std::shared_ptr<Column>>  columns(ncols);
        // 1. allocate & copy meta data
        for(int i = 0;i < lcols; i++) {
            CUDA_CHECK( cudaMallocAsync((void**)&resultData[i], l.columns[i]->bytes * batchSize, stream));
            nullset[i] = std::shared_ptr<NullBitSet>(new NullBitSet(batchSize, stream));// need to be copied also
            columns[i] = l.columns[i]; 
        }
        for(int i = 0;i < rcols; i++) {
            int j = lcols + i;
            CUDA_CHECK( cudaMallocAsync((void**)&resultData[j], r.columns[i]->bytes * batchSize, stream));
            nullset[j] = std::shared_ptr<NullBitSet>(new NullBitSet(batchSize, stream));
            columns[j] = r.columns[i]; 
        }

        // 2. move rows
        for(int i = 0;i < lcols; i++) {
            move_rows_one_col(
                l.columnData[i], 
                resultData[i], 
                pairs, 
                l.nullset[i]->bitset, 
                nullset[i]->bitset, 
                batchSize, l.columns[i]->type, false, stream);
        }
        // cudaDeviceSynchronize();
        for(int i = 0;i < rcols; i++) {
            move_rows_one_col(
                r.columnData[i], 
                resultData[lcols + i], 
                pairs, 
                r.nullset[i]->bitset, 
                nullset[lcols + i]->bitset, 
                batchSize, 
                r.columns[i]->type, true, stream);
        }

        return std::unique_ptr<Batch>(new Batch(resultData, Device::GPU, batchSize, columns, nullset, stream));

    }
    
    void move_rows_one_col(void* src, void* res, uint32_t* pairs, char* isnull, char* out_isnull, const uint32_t batchSize, DataType type, bool isright, cudaStream_t stream) {
        switch (type)
        {
        case DataType::INT:
            YallaSQL::Kernel::lanch_move_rows_join_kernel(static_cast<int*>(src), static_cast<int*>(res), pairs, isnull, out_isnull, batchSize, isright, stream);
            break;
        case DataType::FLOAT:
            YallaSQL::Kernel::lanch_move_rows_join_kernel(static_cast<float*>(src), static_cast<float*>(res), pairs, isnull, out_isnull, batchSize, isright, stream);
            break;
        case DataType::DATETIME:
            YallaSQL::Kernel::lanch_move_rows_join_kernel(static_cast<int64_t*>(src), static_cast<int64_t*>(res), pairs, isnull, out_isnull, batchSize, isright, stream);
            break;
        case DataType::STRING:
            YallaSQL::Kernel::lanch_move_rows_join_kernel(static_cast<YallaSQL::Kernel::String*>(src), static_cast<YallaSQL::Kernel::String*>(res), pairs, isnull, out_isnull, batchSize, isright, stream);
            break;
        
        default:
            break;
        }

    }


    void cacheLeftChild(CacheManager& cacheManager) {
        if(isFirstPass) return;

        BatchID childBatchId = children[0]->next(cacheManager);
        while(childBatchId != 0) {
            left_stack[currStack].push(childBatchId);
            childBatchId = children[0]->next(cacheManager);
        }
        currStack = !currStack;
        isFirstPass = true;
    }
    bool getRightChild(CacheManager& cacheManager) {
        const auto r_batchId = children[1]->next(cacheManager);
        if(r_batchId == 0) {
            rbatch = nullptr;
            return 0;
        }
        // get ownership of child
        rbatch = cacheManager.getBatch(r_batchId);
        if(!rbatch) return 0;
        return 1;
    }
    // should be called only from init
    void initColumns() {
        //! should through error later
        if(children.size() != 2) {
            LOG_ERROR(logger, "Join Operator required 2 children");
            std::cout << "Join Operator required 2 children\n";
            return;
        }
        // get logical operator
        duckdb::LogicalComparisonJoin& castOp = logicalOp.Cast<duckdb::LogicalComparisonJoin>();
        //TODO: may be needed same as expressions of multiple ands
        build_expressions(castOp.conditions);
        // initalize children and get columns to return
        children[0]->init();
        children[1]->init();

        for(auto col: children[0]->columns) {
            columns.push_back(col);
        }
        for(auto col: children[1]->columns) {
            columns.push_back(col);
        }
    }
    std::unique_ptr<our::Expression> getExpressionFromCond(duckdb::JoinCondition& condition) {
        switch (condition.comparison)
        {
        case duckdb::ExpressionType::CONJUNCTION_AND:
        case duckdb::ExpressionType::CONJUNCTION_OR:
            return std::unique_ptr<our::Expression>(new our::ConjuntionExpression(condition));
            break;
        default:
            return std::unique_ptr<our::Expression>(new our::ComparisonJoinExpression(condition));
            break;
        }
    }
    // if there's multiple individual expressions in filter -> and them
    void build_expressions(std::vector<duckdb::JoinCondition>& duckConditions) {
        if(duckConditions.size() == 0) {
            // LOG_ERROR(logger, "No expressions in filter operation");
            // std::cout << "No expressions in filter operation\n";
            throw std::runtime_error("No condition in join operation");
            return;
        }

        if(duckConditions.size() == 1) {
            auto our_expr = getExpressionFromCond(duckConditions[0]);
            expression = std::move(our_expr);
            return;
        }

        // build and tree
        auto left = getExpressionFromCond(duckConditions[0]);
        auto right =  getExpressionFromCond(duckConditions[1]);

        left = std::unique_ptr<our::Expression>(new our::ConjunctionJoinExpression(
            std::move(left),
            std::move(right),
            true
        ));
        for(uint32_t i = 2; i < duckConditions.size();i++) {
            right = getExpressionFromCond(duckConditions[i]);
            left = std::unique_ptr<our::Expression>(new our::ConjunctionJoinExpression(
                std::move(left),
                std::move(right),
                true
            ));
        }

        expression = std::move(left);        
    }
};
}    