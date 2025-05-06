#include "engine/operators/list.hpp"
#include <duckdb/planner/expression/list.hpp>
#include <duckdb/planner/operator/list.hpp>

using namespace YallaSQL;

Operator::~Operator() {
    // since child is unique_ptr we don't need to delete it
    // for(auto& child : children) {
    //     if(child) delete child.get();
    // }
}

std::unique_ptr<Operator> Operator::CreateOperator(const duckdb::LogicalOperator& op, const duckdb::Planner &planner) {
    if(op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto &castOp = op.Cast<duckdb::LogicalGet>();
        // TODO:
        auto &filters = castOp.table_filters.filters;

        return std::unique_ptr<Operator>(new GetOperator(op, planner) );
    } 
    else if(op.type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
        return std::unique_ptr<Operator>(new ProjectionOperator(op, planner) );
    }
    else if(op.type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
        return std::unique_ptr<Operator>(new FilterOperator(op, planner) );
    }
    else if(op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
        return std::unique_ptr<Operator>(new AggregateOperator(op, planner) );
    }
    return nullptr;
    // else if(op.type == duckdb::LogicalOperatorType::LOGICAL_LIMIT) {

    // }
    // else if(op.type == duckdb::LogicalOperatorType::LOGICAL_ORDER_BY) {

    // }
    // else if(op.type == duckdb::LogicalOperatorType::LOGICAL_TOP_N) {

    // }
    // else if(op.type == duckdb::LogicalOperatorType::LOGICAL_DISTINCT) {

    // }
    // -----------------------------
	// Joins
	// -----------------------------
	// LOGICAL_JOIN = 50,
	// LOGICAL_DELIM_JOIN = 51,
	// LOGICAL_COMPARISON_JOIN = 52,
	// LOGICAL_ANY_JOIN = 53,
	// LOGICAL_CROSS_PRODUCT = 54,
	// LOGICAL_POSITIONAL_JOIN = 55,
	// LOGICAL_ASOF_JOIN = 56,
	// LOGICAL_DEPENDENT_JOIN = 57,
    // -----------------------------
	// SetOps
	// -----------------------------
	// LOGICAL_UNION = 75,
	// LOGICAL_EXCEPT = 76,
	// LOGICAL_INTERSECT = 77,
	// LOGICAL_RECURSIVE_CTE = 78,
	// LOGICAL_MATERIALIZED_CTE = 79,
    // else if(op.type == duckdb::LogicalOperatorType::LOGICAL_) {

    // }
}
