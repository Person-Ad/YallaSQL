#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "config.hpp"
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
            std::string valueStr= valueExp.GetValue<std::string>();
            int64_t v = getDateTime(valueStr);
            CUDA_CHECK(cudaMemcpyAsync(value, &v, sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            // CUDA_CHECK(cudaMemcpyToSymbol(value, &v, sizeof(int64_t), 0, cudaMemcpyHostToDevice));
            break;
        }
        case DataType::STRING: {
            std::string v = valueExp.GetValue<std::string>();
            size_t str_size = v.size() + 1; // +1 for null terminator
            CUDA_CHECK(cudaMallocAsync(&value, YallaSQL::MAX_STR_LEN, stream));
            CUDA_CHECK(cudaMemcpyAsync(value, v.c_str(), str_size, cudaMemcpyHostToDevice, stream));
            break;
        }
        }
        CUDA_CHECK_LAST();

    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        CUDA_CHECK( cudaStreamSynchronize(stream) );
        result.result = value;
        result.batchSize = 1;
        result.nullset = nullptr;
        return result;
    }

    ~BoundValueExpression() {
        CUDA_CHECK( cudaFreeAsync(value, stream) );
        CUDA_CHECK( cudaStreamDestroy(stream) );
    }

};

}