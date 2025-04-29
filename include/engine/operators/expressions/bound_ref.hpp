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
        
        Batch* child = arg.batchs[0];
        if(child->location == Device::FS)
            child->moveTo(Device::CPU);
        size_t totalBytes = child->columns[idx]->bytes * child->batchSize;
        //TODO: handle expression output on different devices rather than GPU
        //* current will do gpu only
        void* data;
        CUDA_CHECK( cudaMallocAsync(&data, totalBytes, child->stream ) );


        if(child->location == Device::CPU)  {
            CUDA_CHECK( cudaMemcpyAsync(data, child->getColumn(idx), totalBytes, cudaMemcpyHostToDevice, child->stream) );
        } else {
            CUDA_CHECK( cudaMemcpyAsync(data, child->getColumn(idx), totalBytes, cudaMemcpyDeviceToDevice, child->stream) );
        }

        result.batchSize = child->batchSize;
        result.result = data;
        return result;
    }

};
// aggregate expression Max, Min, Sum, Avg, .... //internal state update inside it 
// binary expression +, -, *, /, % ... // take two batchs or one with scalar or two scalar and return a value
// unary expression -x, !x, ... // take one batch or one scalar and return a value
// constant expression 1, 2, 3, "hello", ... // take no batch and return a value
// bound_ref expression // take one batch and return a batch

}