#pragma once
#include "engine/operators/expressions/expression.hpp"
#include <duckdb/planner/expression/list.hpp>

namespace our {

class BoundValueExpression : public Expression {
public:
    //TODO: make it constant
    // __constant__ char data[8];
    void *value;
    cudaStream_t stream;

    BoundValueExpression(duckdb::Expression &expr): Expression(expr) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        exprType = ExpressionType::BOUND_VALUE;
        is_scalar = true;
        
        auto& valueExp = expr.Cast<duckdb::BoundConstantExpression>().value;
        // allocate space on gpu for value
        CUDA_CHECK( cudaMallocAsync(&value, getDataTypeNumBytes(returnType), stream) );

        switch (returnType) {
        case DataType::INT: {
            int v = valueExp.GetValue<int>();
            CUDA_CHECK(cudaMemcpyAsync(value, &v, sizeof(int), cudaMemcpyHostToDevice, stream));
            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(int), 0, cudaMemcpyHostToDevice));
            break;
        }
        case DataType::FLOAT: {
            float v = valueExp.GetValue<float>();
            CUDA_CHECK(cudaMemcpyAsync(value, &v, sizeof(float), cudaMemcpyHostToDevice, stream));
            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(float), 0, cudaMemcpyHostToDevice));
            break;
        }
        case DataType::DATETIME: {
            float v = valueExp.GetValue<float>();
            CUDA_CHECK(cudaMemcpyAsync(value, &v, sizeof(int64_t), cudaMemcpyHostToDevice, stream));

            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(int64_t), 0, cudaMemcpyHostToDevice));
            break;
        }
        }
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        CUDA_CHECK( cudaStreamSynchronize(stream) );
        result.result = value;
        result.batchSize = 1;
        return result;
    }

    ~BoundValueExpression() {
        CUDA_CHECK( cudaFreeAsync(value, stream) );
        CUDA_CHECK( cudaStreamDestroy(stream) );
    }

};

}