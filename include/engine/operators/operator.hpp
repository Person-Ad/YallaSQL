#pragma once

#include <memory>
#include <vector>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/planner.hpp>
#include "engine/cache_manager.hpp"
#include "logger.hpp"
#include "batch.hpp"

namespace YallaSQL {
class Operator {

public:
    // shcema of batch that return by this operator
    std::vector<std::shared_ptr<Column>> columns;
    // best batchsize of current table
    unsigned int batchSize = 0;
    // is operator is finished
    bool isFinished = false;
protected:
    // children of operator
    std::vector<std::unique_ptr<Operator>> children;
    // logical operator reference 
    const duckdb::LogicalOperator& logicalOp;
    // planner to get bindings
    const duckdb::Planner &planner;
    // logger
    quill::Logger* logger = YallaSQL::getLogger("");
    // initalized flag
    bool isInitalized = false;
    // not like constant value etc..
    bool is_scalar = true;
    bool is_sync = false; // require all children to start

public:
    static std::unique_ptr<Operator> CreateOperator(const duckdb::LogicalOperator& op, const duckdb::Planner &planner);
    
    Operator(const duckdb::LogicalOperator& op, const duckdb::Planner &planner): logicalOp(op), planner(planner) {
        for(auto childIt = op.children.begin(); childIt != op.children.end(); childIt++) {
            auto child = Operator::CreateOperator(**childIt, planner);
            if(child) {
                children.push_back(std::move(child));
            }
        }
    }
    // initalize Operator (eg., opening files etc...)
    virtual void init() = 0;
    // return next batch index or 0 if empty
    virtual BatchID next(CacheManager&) = 0;
    // remove any leftovers
    virtual ~Operator();

protected:
    // initalize all children
    void initChildren() {
        for(auto& child: children) 
            child->init();
        
    }
};

};


