#pragma once
#include "engine/operators/expressions/expression.hpp"
#include <duckdb/planner/expression/list.hpp>

namespace our {

class BoundValueExpression : public Expression {
public:
    //TODO: make it constant
    // __constant__ char data[8];
    void *value;

    BoundValueExpression(duckdb::Expression &expr): Expression(expr) {
        exprType = ExpressionType::BOUND_VALUE;

        auto& valueExp = expr.Cast<duckdb::BoundConstantExpression>().value;

        // allocate space on gpu for value
        CUDA_CHECK( cudaMalloc(&value, getDataTypeNumBytes(returnType)) );

        switch (returnType) {
        case DataType::INT: {
            int v = valueExp.GetValue<int>();
            CUDA_CHECK(cudaMemcpy(value, &v, sizeof(int), cudaMemcpyHostToDevice));
            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(int), 0, cudaMemcpyHostToDevice));
            break;
        }
        case DataType::FLOAT: {
            float v = valueExp.GetValue<float>();
            CUDA_CHECK(cudaMemcpy(value, &v, sizeof(float), cudaMemcpyHostToDevice));
            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(float), 0, cudaMemcpyHostToDevice));
            break;
        }
        case DataType::DATETIME: {
            float v = valueExp.GetValue<float>();
            CUDA_CHECK(cudaMemcpy(value, &v, sizeof(int64_t), cudaMemcpyHostToDevice));

            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(int64_t), 0, cudaMemcpyHostToDevice));
            break;
        }
        }
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        result.result = value;
        result.batchSize = 1;
        return result;
    }

    ~BoundValueExpression() {
        cudaFree(value);
    }

};

}