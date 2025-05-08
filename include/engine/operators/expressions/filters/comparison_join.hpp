#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/comparison_operators_kernel.hpp"
#include "kernels/string_kernel.hpp"
#include "config.hpp"
#include <duckdb/planner/expression/list.hpp>

namespace our {

    enum class JoinCompareType : __uint8_t {
        LE, LEQ, GE, GEQ, EQ, NEQ
    };
    
    [[nodiscard]] inline JoinCompareType getJoinCompareType(duckdb::ExpressionType type) {
        switch(type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return JoinCompareType::EQ;  
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return JoinCompareType::NEQ;  
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return JoinCompareType::LE;  
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return JoinCompareType::GE;  
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return JoinCompareType::LEQ;  
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return JoinCompareType::GEQ;  
        default:
            throw std::runtime_error("Expression Type Not Supported this type");
        }
    } 

class ComparisonJoinExpression: public Expression {
    bool isneg, isjoin;
public:

    std::vector<std::unique_ptr<Expression>> children;
    JoinCompareType compare_type;

    ComparisonJoinExpression(duckdb::Expression &expr, bool isneg = false): Expression(expr), isneg(isneg) {
        exprType = ExpressionType::COMPARISON;
        compare_type = getJoinCompareType(expr.type);

        // get left & right child
        auto& castExpr = expr.Cast<duckdb::BoundComparisonExpression>();
        
        children.reserve(2);
        children.push_back(Expression::createExpression(*castExpr.left));
        children.push_back(Expression::createExpression(*castExpr.right));

        children[0]->table_idx = 0;
        children[1]->table_idx = 1;

    } 
    ComparisonJoinExpression(duckdb::JoinCondition& cond): isneg(false) {
        exprType = ExpressionType::COMPARISON;
        compare_type = getJoinCompareType(cond.comparison);

        isjoin = true;
        
        children.reserve(2);
        children.push_back(Expression::createExpression(*cond.left));
        children.push_back(Expression::createExpression(*cond.right));
        
        children[0]->table_idx = 0;
        children[1]->table_idx = 1;

    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        //! alwayse use left stream
        cudaStream_t& stream = arg.stream;
        size_t lbatchSize = arg.batchs[0]->batchSize;
        size_t rbatchSize = arg.batchs[1]->batchSize;
        
        ExpressionResult res_lhs = children[0]->evaluate(arg);
        ExpressionResult res_rhs = children[1]->evaluate(arg);
        void* lhs = res_lhs.result;
        void* rhs = res_rhs.result; 
        
        int *actualOutSize;
        CUDA_CHECK(cudaMallocAsync((void**)&actualOutSize, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(actualOutSize, 0, sizeof(int), stream));
        // Allocate result memory
        uint32_t* res;
        CUDA_CHECK(cudaMallocAsync(&res, 2*YallaSQL::MAX_ROWS_OUT_JOIN_OP*sizeof(uint32_t), stream));

        switch (compare_type) {
        case JoinCompareType::LE:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::LEOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::LEOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::LEOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::LEOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case JoinCompareType::LEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::LEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::LEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::LEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::LEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case JoinCompareType::GE:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::GEOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::GEOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::GEOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::GEOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case JoinCompareType::GEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::GEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::GEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::GEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::GEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case JoinCompareType::EQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::EQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::EQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::EQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::EQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        case JoinCompareType::NEQ:
            switch (children[0]->returnType) {
                case DataType::INT:
                    YallaSQL::Kernel::launch_outer_join_operators<int, YallaSQL::Kernel::NEQOperator<int>>(
                            static_cast<int*>(rhs), static_cast<int*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::FLOAT:
                    YallaSQL::Kernel::launch_outer_join_operators<float, YallaSQL::Kernel::NEQOperator<float>>(
                            static_cast<float*>(rhs), static_cast<float*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::DATETIME:
                    YallaSQL::Kernel::launch_outer_join_operators<int64_t, YallaSQL::Kernel::NEQOperator<int64_t>>(
                            static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                case DataType::STRING:
                    YallaSQL::Kernel::launch_outer_join_operators<YallaSQL::Kernel::String, YallaSQL::Kernel::NEQOperator<YallaSQL::Kernel::String>>(
                        static_cast<YallaSQL::Kernel::String*>(rhs), static_cast<YallaSQL::Kernel::String*>(lhs), res, actualOutSize, lbatchSize, rbatchSize, stream);
                    break;
                default:
                    throw std::runtime_error("Unsupported data type in comparision");
            }
            break;
        default:
            break;
        }

        CUDA_CHECK(cudaFreeAsync(lhs, stream));
        CUDA_CHECK(cudaFreeAsync(rhs, stream));
        
        result.result = res; 
        //! need ta access using actual block size
        result.d_batchSize = actualOutSize;
        result.batchSize = std::max(res_lhs.batchSize, res_rhs.batchSize);
        //! danger
        // result.nullset = (children[0]->exprType != ExpressionType::BOUND_VALUE) ? res_lhs.nullset : res_rhs.nullset;
        return result;
    }

};
}