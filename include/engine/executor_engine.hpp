#pragma once

#include <memory>
#include <duckdb/planner/planner.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
// time to include our operators :

#include "engine/operators/get_operator.hpp"
#include "engine/cache_manager.hpp"

class ExecutorEngine {

public:
    void execute(const duckdb::LogicalOperator& logicalPlan, const duckdb::Planner& planner);

private:
    // void getPhysicalPlan(const duckdb::LogicalOperator& op, CacheManager &cacheManager);

};
