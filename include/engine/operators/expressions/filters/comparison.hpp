#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/comparison_operators_kernel.hpp"
#include "kernels/string_kernel.hpp"

#include <duckdb/planner/expression/list.hpp>

namespace our {

    enum class CompareType : __uint8_t {
        LE, LEQ, GE, GEQ, EQ, NEQ
    };
    [[nodiscard]] inline CompareType getCompareType(duckdb::Expression &expr) {
        switch(expr.type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return CompareType::EQ;  
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return CompareType::NEQ;  
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return CompareType::LE;  
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return CompareType::GE;  
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return CompareType::LEQ;  
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return CompareType::GEQ;  
        default:
            throw std::runtime_error("Expression Type Not Supported: " + expr.ToString());
        }
    } 

class ComparisonExpression: public Expression {
    bool isneg;
public:

    std::vector<std::unique_ptr<Expression>> children;
    CompareType compare_type;

    ComparisonExpression(duckdb::Expression &expr, bool isneg = false): Expression(expr), isneg(isneg) {
        exprType = ExpressionType::COMPARISON;
        compare_type = getCompareType(expr);

        // get left & right child
        auto& castExpr = expr.Cast<duckdb::BoundComparisonExpression>();
        
        children.reserve(2);
        children.push_back(Expression::createExpression(*castExpr.left));
        children.push_back(Expression::createExpression(*castExpr.right));
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;

        cudaStream_t& stream = arg.batchs[0]->stream;
        size_t batchSize = arg.batchs[0]->batchSize;

        ExpressionResult res_lhs = children[0]->evaluate(arg);
        ExpressionResult res_rhs = children[1]->evaluate(arg);
        void* lhs = res_lhs.result;
        void* rhs = res_rhs.result; 

        YallaSQL::Kernel::OperandType t_lhs = res_lhs.batchSize == 1 ? YallaSQL::Kernel::OperandType::SCALAR : YallaSQL::Kernel::OperandType::VECTOR;
        YallaSQL::Kernel::OperandType t_rhs = res_rhs.batchSize == 1 ? YallaSQL::Kernel::OperandType::SCALAR : YallaSQL::Kernel::OperandType::VECTOR;
        // Allocate result memory
        bool* res;
        CUDA_CHECK(cudaMallocAsync(&res, batchSize, stream));

        switch (compare_type) {
        case CompareType::LE:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::LEOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::LEOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::LEOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::LEOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case CompareType::LEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::LEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::LEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::LEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::LEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case CompareType::GE:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::GEOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::GEOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::GEOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::GEOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case CompareType::GEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::GEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::GEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::GEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::GEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case CompareType::EQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::EQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::EQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::EQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::EQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case CompareType::NEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_conditional_operators<int, YallaSQL::Kernel::NEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_conditional_operators<float, YallaSQL::Kernel::NEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_conditional_operators<int64_t, YallaSQL::Kernel::NEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_conditional_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::NEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), t_rhs, t_lhs, res, batchSize, stream, isneg);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        default:
            break;
        }


        if(children[0]->exprType != ExpressionType::BOUND_VALUE)
            CUDA_CHECK(cudaFreeAsync(lhs, stream));
        if(children[1]->exprType != ExpressionType::BOUND_VALUE)
            CUDA_CHECK(cudaFreeAsync(rhs, stream));
        
        result.batchSize = std::max(res_lhs.batchSize, res_rhs.batchSize);
        result.result = res; 
        result.nullset = (children[0]->exprType != ExpressionType::BOUND_VALUE) ? res_lhs.nullset : res_rhs.nullset;
        return result;
    }

};
}