#pragma once
#include "engine/operators/expressions/expression.hpp"
#include <duckdb/planner/expression/list.hpp>


namespace our {

class CastExpression : public Expression {
    std::unique_ptr<Expression> child;
    DataType childType;
public:
    CastExpression(duckdb::Expression &expr): Expression(expr) {
        exprType = ExpressionType::CAST;

        child = Expression::createExpression(
            *expr.Cast<duckdb::BoundCastExpression>().child
        );

        childType = child->returnType;
    }
    
    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;

        auto values = child->evaluate(arg).result;
        
        void* res;
        // CUDA_CHECK( CudaMalloc(&res, ) )
        // if()

        return result;
    }

};

}