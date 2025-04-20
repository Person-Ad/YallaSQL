#pragma once

#include <memory>
#include <duckdb/planner/planner.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
// time to include our operators :

#include "engine/operators/get_operator.hpp"
#include "engine/cache_manager.hpp"

class ExecutorEngine {

public:
    void execute(const duckdb::LogicalOperator& logicalPlan, duckdb::Binder &binder) {
        CacheManager cacheManager;
        getPhysicalPlan(logicalPlan, binder, cacheManager);
    }

private:
    void getPhysicalPlan(const duckdb::LogicalOperator& op, duckdb::Binder &binder, CacheManager &cacheManager) {
        auto opType = op.type;
        auto numChild = op.children.size();
        auto numExpressions = op.expressions.size();

        if(opType == LogicalOperatorType::LOGICAL_PROJECTION) {
            const auto& castOp = op.Cast<LogicalProjection>();
            
        } 
        else if (opType == LogicalOperatorType::LOGICAL_GET) {
            // const LogicalGet& castOp = op.Cast<LogicalGet>();
            YallaSQL::GetOperator ourOP;
            ourOP.init(op);
            BatchID batchId = ourOP.next(cacheManager);
            Batch* batch = cacheManager.getBatch(batchId, Device::CPU);
            batch->print();
            // auto table = castOp.GetTable().get()->name;
            // const auto& colIds = castOp.GetColumnIds();
            // for(auto col_id: colIds) {
            //     auto colName = castOp.GetColumnName(col_id);
            //     std::cout << table << " " << colName << "\n";
            // }

        }
        else if (opType == LogicalOperatorType::LOGICAL_CHUNK_GET) {
            // const auto& castOp = op.Cast<LogicalGet>();

        }

        // for(int i = 0;i < numExpressions;i++) {
        //     auto expType = op.expressions[i]->type;
        //     if(expType == ExpressionType::BOUND_COLUMN_REF) {
        //         auto &castExpr = op.expressions[i]->Cast<BoundColumnRefExpression>();
        //         duckdb::idx_t col = castExpr.binding.column_index;
        //         duckdb::idx_t table = castExpr.binding.table_index;
        //         std::cout << col << " " << table << "\n";
        //     }
        //     // duckdb::ExpressionIterator::EnumerateExpression()
        // }
        if(op.children.size() == 0) return;
        for(int i = 0;i < numChild;i++){
            if(op.children[i])
                getPhysicalPlan(*op.children[i], binder, cacheManager);
        }
            
    }


};
