#include "engine/operators/expressions/list.hpp"


namespace our {

    std::unique_ptr<Expression> Expression::createExpression(duckdb::Expression &expr, bool isneg, bool isjoin) {
        // if(isjoin) {
        // switch(expr.type) {
        //     case duckdb::ExpressionType::COMPARE_EQUAL:
        //     case duckdb::ExpressionType::COMPARE_NOTEQUAL:
        //     case duckdb::ExpressionType::COMPARE_LESSTHAN:
        //     case duckdb::ExpressionType::COMPARE_GREATERTHAN:
        //     case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
        //     case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
        //         return std::unique_ptr<Expression>(new ComparisonJoinExpression(expr));  
        //     case duckdb::ExpressionType::CONJUNCTION_AND:
        //     case duckdb::ExpressionType::CONJUNCTION_OR:
        //         return std::unique_ptr<Expression>(new ConjuntionJoinExpression(expr));  
        // }
        // }
        switch(expr.type) {
            case duckdb::ExpressionType::BOUND_REF:
                return std::unique_ptr<Expression>(new BoundRefExpression(expr));
            case duckdb::ExpressionType::VALUE_CONSTANT:
                return std::unique_ptr<Expression>(new BoundValueExpression(expr));
            case duckdb::ExpressionType::BOUND_FUNCTION:
                return std::unique_ptr<Expression>(new BoundFuncExpression(expr));
            case duckdb::ExpressionType::CAST:
            case duckdb::ExpressionType::OPERATOR_CAST:
                return std::unique_ptr<Expression>(new CastExpression(expr));
            case duckdb::ExpressionType::COMPARE_EQUAL:
            case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            case duckdb::ExpressionType::COMPARE_LESSTHAN:
            case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                return std::unique_ptr<Expression>(new ComparisonExpression(expr));  
            case duckdb::ExpressionType::CONJUNCTION_AND:
            case duckdb::ExpressionType::CONJUNCTION_OR:
                return std::unique_ptr<Expression>(new ConjuntionExpression(expr));  
            case duckdb::ExpressionType::OPERATOR_NOT:
                return createExpression(*expr.Cast<duckdb::BoundComparisonExpression>().left, isneg);  
            case duckdb::ExpressionType::BOUND_AGGREGATE:
            case duckdb::ExpressionType::AGGREGATE:
                return std::unique_ptr<Expression>(new AggregateExpression(expr));
                break;
            default:
                throw std::runtime_error("Expression Type Not Supported: " + expr.ToString());
        }
    }
}

/*
// IN operator [left IN (right1, right2, ...)]
COMPARE_IN = 35,
// NOT IN operator [left NOT IN (right1, right2, ...)]
COMPARE_NOT_IN = 36,
// IS DISTINCT FROM operator
COMPARE_DISTINCT_FROM = 37,

COMPARE_BETWEEN = 38,
COMPARE_NOT_BETWEEN = 39,

// IS NOT DISTINCT FROM operator
COMPARE_NOT_DISTINCT_FROM = 40,
// compare final boundary
COMPARE_BOUNDARY_END = COMPARE_NOT_DISTINCT_FROM,
*/