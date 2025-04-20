#pragma once

#include <memory>
#include <vector>
#include <duckdb/planner/logical_operator.hpp>
#include "engine/cache_manager.hpp"
#include "batch.cu"

namespace YallaSQL {
class Operator {

private:
    // children of operator
    std::vector<std::unique_ptr<Operator>> children;
    // shcema of batch that return by this operator
    std::vector<DataType> schema;
    // byte size of values return
    size_t byteSize;
public:
    // initalize Operator (eg., opening files etc...)
    virtual void init(const duckdb::LogicalOperator& ) = 0;
    // return next batch index or 0 if empty
    virtual BatchID next(CacheManager&) = 0;
    // remove any leftovers
    virtual ~Operator() = 0;
};

Operator::~Operator() {}
};


