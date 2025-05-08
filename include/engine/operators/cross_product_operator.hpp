#pragma once

#include "db/table.hpp"
#include "engine/operators/operator.hpp"
#include "kernels/string_kernel.hpp"

#include <duckdb/planner/operator/list.hpp>
#include <duckdb/parser/expression/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include "engine/operators/expressions/bound_ref.hpp"
#include <memory>

#include "kernels/cross_product_kernel.hpp"

namespace YallaSQL {

class CrossProductOperator final: public Operator {
    const size_t MAX_RIGHT_SZ = 256;
private:
    std::vector<std::unique_ptr<our::Expression>> expressions;
    std::unique_ptr<Batch> rbatch = nullptr;
    std::stack<BatchID> left_stack[2];
    int rrow_idx = 0;
    bool isFirstPass = false;
    bool currStack = 0;

public:
    // inherit from operator
    using Operator::Operator; 
    
    void init() override  {
        if(isInitalized) return;
        // get columns & projections from logical operator
        initColumns();
        // update state
        isInitalized = true;
    }
    // get children batch & remove unnecessary columns
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
            if(left_stack[currStack].empty()) 
                currStack = !currStack;
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
        auto resBatch = cross_product_batchs(*lbatch, *rbatch);
        BatchID leftBactchId = cacheManager.putBatch(std::move(lbatch));
        BatchID resBatchId = cacheManager.putBatch(std::move(resBatch));
        if(rrow_idx < rbatch->batchSize) {
            left_stack[currStack].push(leftBactchId);
        }
        else {
            left_stack[!currStack].push(leftBactchId);
            rrow_idx = 0;
        }

        return resBatchId;
    }

private:

    std::unique_ptr<Batch> cross_product_batchs(Batch& l, Batch& r) {
        // size_t nrows = r.batchSize * l.batchSize; 
        int rrows = std::min(MAX_RIGHT_SZ, r.batchSize - rrow_idx);
        size_t nrows = rrows * l.batchSize; 
        size_t ncols = l.columns.size() + r.columns.size();

        cudaStreamSynchronize(r.stream);
        cudaStreamSynchronize(l.stream);
        cudaStream_t stream; 
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<void*> resultData(ncols);
        std::vector<std::shared_ptr<NullBitSet>> nullset(ncols);
        std::vector<std::shared_ptr<Column>>  columns(ncols);
        // 1. allocate & copy meta data
        for(int i = 0;i < l.columns.size(); i++) {
            CUDA_CHECK( cudaMallocAsync((void**)&resultData[i], l.columns[i]->bytes * nrows, stream));
            nullset[i] = std::shared_ptr<NullBitSet>(new NullBitSet(nrows, stream));// need to be copied also
            columns[i] = l.columns[i]; 
        }
        for(int i = 0;i < r.columns.size(); i++) {
            int j = l.columns.size() + i;
            CUDA_CHECK( cudaMallocAsync((void**)&resultData[j], r.columns[i]->bytes * nrows, stream));
            nullset[j] = std::shared_ptr<NullBitSet>(new NullBitSet(nrows, stream));
            columns[j] = r.columns[i]; 
        }

        // 2. cross product rows
        for(int i = 0;i < l.columns.size(); i++) {
            YallaSQL::Kernel::launch_cross_product_type(l.columnData[i], resultData[i], l.nullset[i]->bitset, nullset[i]->bitset, columns[i]->type, l.batchSize, rrows, stream, true);
        }
        for(int i = 0;i < r.columns.size(); i++) {
            int j = l.columns.size() + i;
            YallaSQL::Kernel::launch_cross_product_one_type(r.columnData[i], resultData[j], rrow_idx, r.nullset[i]->bitset, nullset[j]->bitset, columns[j]->type, l.batchSize, rrows, stream, false);
        }
        rrow_idx+=rrows;
        // 3. create batch and return it
        return std::unique_ptr<Batch>(new Batch(resultData, Device::GPU, nrows, columns, nullset, stream));
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
    // initiate schema of operator
    // should be called only from init
    void initColumns() {
        //! should through error later
        if(children.empty()) {
            LOG_ERROR(logger, "Projection Operator has no children");
            std::cout << "Projection Operator has no children\n";
            return;
        }
        // get logical operator
        children[0]->init();
        children[1]->init();

        for(auto col: children[0]->columns) {
            columns.push_back(col);
        }
        for(auto col: children[1]->columns) {
            columns.push_back(col);
        }
    }
};
} // YallaSQL