#include "engine/operators/expressions/list.hpp"


namespace our {

    std::unique_ptr<Expression> Expression::createExpression(duckdb::Expression &expr) {
        switch(expr.type) {
            case duckdb::ExpressionType::BOUND_REF:
                return std::unique_ptr<Expression>(new BoundRefExpression(expr));
            case duckdb::ExpressionType::VALUE_CONSTANT:
                return std::unique_ptr<Expression>(new BoundValueExpression(expr));
            case duckdb::ExpressionType::BOUND_FUNCTION:
                return std::unique_ptr<Expression>(new BoundFuncExpression(expr));
            case duckdb::ExpressionType::CAST:
                return std::unique_ptr<Expression>(new CastExpression(expr));
            default:
                throw std::runtime_error("Expression Type Not Supported: " + expr.ToString());
        }
    }
}