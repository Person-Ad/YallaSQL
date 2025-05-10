#pragma once

#include <memory>
#include <duckdb/planner/planner.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/column_binding_map.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>

// time to include our operators :
#include "engine/operators/list.hpp"
#include "engine/cache_manager.hpp"

class ExecutorEngine {

public:
    void execute(duckdb::LogicalOperator& logicalPlan, const duckdb::Planner& planner, std::string name = "");

private:
    // void getPhysicalPlan(const duckdb::LogicalOperator& op, CacheManager &cacheManager);
    void saveBufferedBatchs(std::vector<BatchID>&, CacheManager &cacheManager, CsvWriter & csvWriter) ;
};
