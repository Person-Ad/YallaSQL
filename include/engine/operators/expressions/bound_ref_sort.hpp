#pragma once
#include <memory>
#include "engine/operators/expressions/expression.hpp"
#include "kernels/radix_sort_kernel.hpp"
// #include ""
namespace our {

class BoundRefSortExpression  {
public:
    uint64_t idx; // index of column in children
    DataType srcType;
    DataType returnType;

    BoundRefSortExpression(duckdb::Expression &expr){
        returnType = getDataTypeFromDuck( expr.return_type.id() );

        idx = expr.Cast<duckdb::BoundReferenceExpression>().index;  

        srcType = returnType;
        if(srcType == DataType::FLOAT) {
            returnType = DataType::INT;
        }

    } 

    ExpressionResult evaluate(Batch& batch ) {
        ExpressionResult result;
        
        if(batch.location == Device::FS)
            batch.moveTo(Device::CPU);
        size_t totalBytes = batch.columns[idx]->bytes * batch.batchSize;
        //TODO: handle expression output on different devices rather than GPU
        //* current will do gpu only
        void* buffer, *data;
        CUDA_CHECK( cudaMallocAsync(&buffer, totalBytes, batch.stream ) );
        CUDA_CHECK( cudaMallocAsync(&data, totalBytes, batch.stream ) ); // since I convert it to same


        if(batch.location == Device::CPU)  {
            CUDA_CHECK( cudaMemcpyAsync(buffer, batch.getColumn(idx), totalBytes, cudaMemcpyHostToDevice, batch.stream) );
        } else {
            CUDA_CHECK( cudaMemcpyAsync(buffer, batch.getColumn(idx), totalBytes, cudaMemcpyDeviceToDevice, batch.stream) );
        }
        switch (srcType) {
        case DataType::INT:
            YallaSQL::Kernel::launch_int_to_uint32(static_cast<int*>(buffer), static_cast<uint32_t*>(data), batch.batchSize);
            CUDA_CHECK(cudaFreeAsync(buffer, batch.stream));
            break;
        case DataType::FLOAT:
            YallaSQL::Kernel::launch_float_to_uint32(static_cast<float*>(buffer), static_cast<uint32_t*>(data), batch.batchSize);
            CUDA_CHECK(cudaFreeAsync(buffer, batch.stream));
            break;
        case DataType::STRING:
        case DataType::DATETIME:
            data = buffer;
        default:
            break;
        }

        result.batchSize = batch.batchSize;
        result.result = data;
        result.nullset = batch.nullset[idx];
        return result;
    }

};
}