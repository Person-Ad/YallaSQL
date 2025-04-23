#pragma once

#include "db/table.hpp"
#include "engine/operators/operator.hpp"

#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/parser/expression/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include <memory>

namespace YallaSQL {

class ProjectionOperator final: public Operator {

private:
    std::unordered_map<uint32_t, std::pair<uint32_t, std::string>> projections; // old index: alias, curr indexs
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
        if(isFinished && children.empty()) {isFinished = true; return 0;}
        // get batchId from children
        const auto batchId = children[0]->next(cacheManager);
        if(batchId == 0) {isFinished = true; return 0;}
        // get ownership of batch
        auto batch = cacheManager.getBatch(batchId);
        // 1. get new data order
        std::vector<void*> columnData (columns.size());
        for (const auto& [oldIdx, metadata]: projections) {
            columnData[metadata.first] = batch->getColumn(oldIdx); // get old index
        }
        // 2. remove columns
        uint32_t numRemovedCols = 0;
        for (uint32_t i = 0; i < batch->columns.size(); ++i) {
            if (!projections.contains(i))
                batch->removeColumn(i - numRemovedCols++);

        }
        // 3. update batch
        batch->updateColumns(columnData, columns);
        // return ownership to manager

        return cacheManager.putBatch(std::move(batch));
    }
    // delete data
    ~ProjectionOperator() override  {
        
    }
private:
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
        const auto& castOp = logicalOp.Cast<LogicalProjection>();
        // get children schema
        const auto& child = children[0];
        child->init();
        const auto& childColumns = child->columns;
        // loop on expression getting indexes & aliases
        uint32_t index = 0;
        for (const auto& expr : castOp.expressions) {
            if(expr->type == ExpressionType::BOUND_REF) {
//                auto& boundRef = expr->Cast<BoundColumnRefExpression>();
                auto& boundRef = expr->Cast<BoundReferenceExpression>();
                auto& oldIndex = boundRef.index;
                auto& alias = boundRef.alias;
                // store projection & column
                projections[oldIndex] = {index++, alias};
                columns.push_back(std::shared_ptr<Column> (
                    new Column(alias, childColumns[oldIndex]->type, childColumns[oldIndex]->isPk, childColumns[oldIndex]->isFk)
                ));
            } else {
                LOG_ERROR(logger, "Projection Operator Found new Expression : {} -> {}", static_cast<int>(expr->type), expr->ToString());
                std::cout << "Projection Operator Found new Expression : " << static_cast<int>(expr->type)  << "\n" << expr->ToString() << "\n";
            }
        }
    }
};
} // YallaSQL