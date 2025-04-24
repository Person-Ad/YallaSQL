#pragma once
#include <memory>
#include "utils/macros.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "enums/data_type.hpp"
#include "enums/device.hpp"
#include "batch.hpp"
#include "batch.hpp"
// #include ""
namespace our {


enum class ExpressionType : __uint8_t {
    BOUND_REF,
};


struct ExpressionResult {
    // for async values +, - ....
    void* result;
    // for groups
    std::vector<void*> groups;
    // for aggregates with 1 val
    void* scalarValue;
};

struct ExpressionArg {
    std::vector<Batch*> batchs;
};

class Expression {
public:
    std::string alias;
    DataType returnType;
    std::shared_ptr<Column> column;
    bool is_sync  = false; // if need all children to output Like Max, Min, ....
    bool is_group = false; // have group
    ExpressionType exprType;

    Expression(duckdb::Expression &expr) {
        alias = expr.alias;
        returnType = getDataTypeFromDuck( expr.return_type.id() );

        column = std::shared_ptr<Column>(new Column(alias, returnType));
    }
    // evaluate expression
    virtual ExpressionResult evaluate(ExpressionArg arg) = 0;

};

class BoundRefExpression : public Expression {
public:
    uint64_t idx; // index of column in children

    BoundRefExpression(duckdb::Expression &expr): Expression(expr) {
        idx = expr.Cast<duckdb::BoundReferenceExpression>().index;  
        exprType = ExpressionType::BOUND_REF;
    } 

    ExpressionResult evaluate(ExpressionArg arg) {
        if(arg.batchs.empty()) {
            throw std::runtime_error("No Children Found For BoundRefExpression");
        }

        Batch* child = arg.batchs[0];
        if(child->location == Device::FS)
            child->moveTo(Device::CPU);
        // cpy data
        //TODO: handle expression output on different devices rather than GPU
        //* current will do gpu only
        void* data;
        CUDA_CHECK( cudaMalloc(&data, child->columns[idx]->bytes * child->batchSize) );


        if(child->location == Device::CPU)  {
            CUDA_CHECK( cudaMemcpy(data, child->getColumn(idx), child->columns[idx]->bytes * child->batchSize, cudaMemcpyHostToDevice) );
        } else {
            CUDA_CHECK( cudaMemcpy(data, child->getColumn(idx), child->columns[idx]->bytes * child->batchSize, cudaMemcpyDeviceToDevice) );
        }

        ExpressionResult result;
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