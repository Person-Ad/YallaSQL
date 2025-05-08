#pragma once
#include <memory>
#include "engine/operators/expressions/expression.hpp"
// #include ""
namespace our {

class BoundRefExpression : public Expression {
public:
    uint64_t idx; // index of column in children

    BoundRefExpression(duckdb::Expression &expr): Expression(expr) {
        idx = expr.Cast<duckdb::BoundReferenceExpression>().index;  
        exprType = ExpressionType::BOUND_REF;
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        if(arg.batchs.empty()) {
            throw std::runtime_error("No Children Found For BoundRefExpression");
        }
        ExpressionResult result;
        
        Batch* child = table_idx == -1 ? arg.batchs[0] : arg.batchs[table_idx];
        if(child->location == Device::FS)
        child->moveTo(Device::CPU);
        size_t totalBytes = child->columns[idx]->bytes * child->batchSize;
        cudaStream_t stream = arg.stream ? arg.stream : child->stream;
        //TODO: handle expression output on different devices rather than GPU
        //* current will do gpu only
        void* data;
        CUDA_CHECK( cudaMallocAsync(&data, totalBytes, child->stream ) );


        if(child->location == Device::CPU)  {
            CUDA_CHECK( cudaMemcpyAsync(data, child->getColumn(idx), totalBytes, cudaMemcpyHostToDevice, stream) );
        } else {
            CUDA_CHECK( cudaMemcpyAsync(data, child->getColumn(idx), totalBytes, cudaMemcpyDeviceToDevice, stream) );
        }

        result.batchSize = child->batchSize;
        result.result = data;
        result.nullset = child->nullset[idx];
        return result;
    }

};
}