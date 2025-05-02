#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/binary_operators_kernel.hpp"

namespace our {

    enum class FunctionType : __uint8_t {
        // Binary Operators
        ADD,
        MINUS,
        MUL,
        DIV,
        REM, // %
        // Uniary Operators
        // filters
    };
    [[nodiscard]] inline FunctionType getFunctionType(std::string str) {
        if(str == "+") return FunctionType::ADD;
        if(str == "-") return FunctionType::MINUS;
        if(str == "*") return FunctionType::MUL;
        if(str == "/") return FunctionType::DIV;
        return FunctionType::REM;
    }    

class BoundFuncExpression: public Expression {
public:

    std::vector<std::unique_ptr<Expression>> children;
    FunctionType function_type;

    BoundFuncExpression(duckdb::Expression &expr): Expression(expr) {
        exprType = ExpressionType::BOUND_FUNC;

        auto& castExpr = expr.Cast<duckdb::BoundFunctionExpression>();

        function_type = getFunctionType(castExpr.function.name);

        children.reserve(castExpr.children.size());
        for(auto& child: castExpr.children) {
            children.push_back( Expression::createExpression(*child) );
        }
        //TODO: recheck is_scalar or not

    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        if (children.size() != 2) {
            throw std::runtime_error("Binary operator requires exactly 2 children");
        }

        cudaStream_t& stream = arg.batchs[0]->stream;   // 
        size_t batchSize = arg.batchs[0]->batchSize; // Assume ExpressionArg provides batchSize

        ExpressionResult res_lhs = children[0]->evaluate(arg);
        ExpressionResult res_rhs = children[1]->evaluate(arg);

        void* lhs = res_lhs.result;
        void* rhs = res_rhs.result;

        YallaSQL::Kernel::OperandType t_lhs = res_lhs.batchSize == 1 ? YallaSQL::Kernel::OperandType::SCALAR : YallaSQL::Kernel::OperandType::VECTOR;
        YallaSQL::Kernel::OperandType t_rhs = res_rhs.batchSize == 1 ? YallaSQL::Kernel::OperandType::SCALAR : YallaSQL::Kernel::OperandType::VECTOR;
        // Allocate result memory
        void* res;
        CUDA_CHECK(cudaMallocAsync(&res, getDataTypeNumBytes(returnType) * batchSize, stream));


        switch (function_type) {
            case FunctionType::ADD:
                switch (returnType) {
                    case DataType::INT:
                        YallaSQL::Kernel::launch_binary_operators<int, YallaSQL::Kernel::AddOperator<int>>(
                                static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, static_cast<int*>(res), batchSize, stream);
                        break;
                    case DataType::FLOAT:
                        YallaSQL::Kernel::launch_binary_operators<float, YallaSQL::Kernel::AddOperator<float>>(
                                static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, static_cast<float*>(res), batchSize, stream);
                        break;
                    case DataType::DATETIME:
                        YallaSQL::Kernel::launch_binary_operators<int64_t, YallaSQL::Kernel::AddOperator<int64_t>>(
                                static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, static_cast<int64_t*>(res), batchSize, stream);
                        break;
                    default:
                        throw std::runtime_error("Unsupported data type");
                }
                break;
            case FunctionType::MINUS:
                switch (returnType) {
                    case DataType::INT:
                        YallaSQL::Kernel::launch_binary_operators<int, YallaSQL::Kernel::MinusOperator<int>>(
                                static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, static_cast<int*>(res), batchSize, stream);
                        break;
                    case DataType::FLOAT:
                        YallaSQL::Kernel::launch_binary_operators<float, YallaSQL::Kernel::MinusOperator<float>>(
                                static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, static_cast<float*>(res), batchSize, stream);
                        break;
                    case DataType::DATETIME:
                        YallaSQL::Kernel::launch_binary_operators<int64_t, YallaSQL::Kernel::MinusOperator<int64_t>>(
                                static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, static_cast<int64_t*>(res), batchSize, stream);
                        break;
                    default:
                        throw std::runtime_error("Unsupported data type");
                }
                break;
            case FunctionType::MUL:
                switch (returnType) {
                    case DataType::INT:
                        YallaSQL::Kernel::launch_binary_operators<int, YallaSQL::Kernel::MulOperator<int>>(
                                static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, static_cast<int*>(res), batchSize, stream);
                        break;
                    case DataType::FLOAT:
                        YallaSQL::Kernel::launch_binary_operators<float, YallaSQL::Kernel::MulOperator<float>>(
                                static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, static_cast<float*>(res), batchSize, stream);
                        break;
                    case DataType::DATETIME:
                        YallaSQL::Kernel::launch_binary_operators<int64_t, YallaSQL::Kernel::MulOperator<int64_t>>(
                                static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, static_cast<int64_t*>(res), batchSize, stream);
                        break;
                    default:
                        throw std::runtime_error("Unsupported data type");
                }
                break;
            case FunctionType::DIV:
                switch (returnType) {
                    case DataType::INT:
                        YallaSQL::Kernel::launch_binary_operators<int, YallaSQL::Kernel::DivOperator<int>>(
                                static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, static_cast<int*>(res), batchSize, stream);
                        break;
                    case DataType::FLOAT:
                        YallaSQL::Kernel::launch_binary_operators<float, YallaSQL::Kernel::DivOperator<float>>(
                                static_cast<float*>(rhs), static_cast<float*>(lhs), t_rhs, t_lhs, static_cast<float*>(res), batchSize, stream);
                        break;
                    case DataType::DATETIME:
                        YallaSQL::Kernel::launch_binary_operators<int64_t, YallaSQL::Kernel::DivOperator<int64_t>>(
                                static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, static_cast<int64_t*>(res), batchSize, stream);
                        break;
                    default:
                        throw std::runtime_error("Unsupported data type");
                }
                break;
            case FunctionType::REM:
                switch (returnType) {
                    case DataType::INT:
                        YallaSQL::Kernel::launch_binary_operators<int, YallaSQL::Kernel::RemOperator<int>>(
                                static_cast<int*>(rhs), static_cast<int*>(lhs), t_rhs, t_lhs, static_cast<int*>(res), batchSize, stream);
                        break;
                    case DataType::DATETIME:
                        YallaSQL::Kernel::launch_binary_operators<int64_t, YallaSQL::Kernel::RemOperator<int64_t>>(
                                static_cast<int64_t*>(rhs), static_cast<int64_t*>(lhs), t_rhs, t_lhs, static_cast<int64_t*>(res), batchSize, stream);
                        break;
                    default:
                        throw std::runtime_error("Unsupported data type");
                }
                break;
            default:
                throw std::runtime_error("Unsupported function type");
        }


        CUDA_CHECK(cudaFreeAsync(lhs, stream));
        CUDA_CHECK(cudaFreeAsync(rhs, stream));

        result.batchSize = std::max(res_lhs.batchSize, res_rhs.batchSize);
        result.result = res;
        return result;
    }

};
}