#pragma once

#include "db/table.hpp"
#include "engine/operators/operator.hpp"

#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/parser/expression/list.hpp>
#include <duckdb/planner/expression/list.hpp>
#include "engine/operators/expressions/bound_ref.hpp"
#include <memory>

namespace YallaSQL {

class ProjectionOperator final: public Operator {

private:
    std::unordered_map<uint32_t, std::pair<uint32_t, std::string>> projections; // old index: alias, curr indexs
    std::vector<std::unique_ptr<our::Expression>> expressions;
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
        const auto childBatchId = children[0]->next(cacheManager);
        if(childBatchId == 0) {isFinished = true; return 0;}
        // get ownership of child
        auto childBatch = cacheManager.getBatch(childBatchId);
        size_t batchSize = childBatch->batchSize;
        childBatch->moveTo(Device::GPU);
        // pass batch to references & store them
        std::vector<void*> resultData(columns.size());
        // pass to expression
        our::ExpressionArg arg {};
        arg.batchs.push_back(childBatch.get());
        // evaluate expression
        uint expIdx = 0;
        for(auto& expr: expressions) {
            our::ExpressionResult res = expr->evaluate(arg);
            resultData[expIdx++] = res.result;
        }

        auto batch = std::unique_ptr<Batch>(new Batch( resultData, Device::GPU,  batchSize, columns));
        //! testing
        batch->moveTo(Device::CPU);
        // return new batch
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
        
        uint32_t index = 0;
        for (auto& expr : castOp.expressions) {
            auto our_expr = our::Expression::createExpression(*expr);
            columns.push_back(our_expr->column);
            expressions.push_back(std::move(our_expr));
/*            if(expr->type == ExpressionType::BOUND_REF) {
                auto our_expr = std::unique_ptr<our::Expression>(
                    new our::BoundRefExpression(*expr)
                );
                columns.push_back(our_expr->column);
                expressions.push_back(std::move(our_expr));

            }
            else if(expr->type == ExpressionType::BOUND_FUNCTION) {
                auto& castExpr = expr->Cast<BoundFunctionExpression>();
                for(auto& exprCC: castExpr.children) {
                    if(exprCC->type == ExpressionType::VALUE_CONSTANT) {
                        auto& castExprCC = expr->Cast<BoundConstantExpression>();
                        auto value = castExprCC.value;

                    }
                }
                std::cout << castExpr.function.name << " " << castExpr.is_operator << "\n";
                std::cout << castExpr.ToString() << "\n";
            }
            else {
                LOG_ERROR(logger, "Projection Operator Found new Expression : {} -> {}", static_cast<int>(expr->type), expr->ToString());
                std::cout << "Projection Operator Found new Expression : " << static_cast<int>(expr->type)  << "\n" << expr->ToString() << "\n";
            }*/
        }
    }
};
} // YallaSQL