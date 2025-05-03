#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/reduction_kernel.hpp"
#include "kernels/binary_operators_kernel.hpp"
namespace our {

    enum class AggType : __uint8_t {
        MAX, MIN, SUM, AVG, COUNT
    };

    [[nodiscard]] inline AggType getAggType(std::string str) {
        if(str == "max") return AggType::MAX;
        if(str == "min") return AggType::MIN;
        if(str == "sum") return AggType::SUM;
        return AggType::AVG;
    } 
    
class AggregateExpression: public Expression {
    //THOSE ARE MORE STABLE TO USE THAN CUDAMEMSET || Op.initial
    int initial_value_int; // for datetime to not overflow in memset since it take int val
    int64_t initial_value_date; // for datetime to not overflow in memset since it take int val
    float initial_value_float; // for float to not overflow in memset since it take int val
    double initial_value_double; // for float to not overflow in memset since it take int val
    float *tempAcc = nullptr;
    std::vector<int> batchSizes;
    void* accumlator;
public:

    std::vector<std::unique_ptr<Expression>> children;
    AggType agg_type;

    AggregateExpression(duckdb::Expression &expr): Expression(expr) {
        exprType = ExpressionType::AGGREGRATE;

        auto& castExpr = expr.Cast<duckdb::BoundAggregateExpression>();
        agg_type = getAggType(castExpr.function.name);

        children.reserve(castExpr.children.size());
        for(auto& child: castExpr.children) {
            children.push_back( Expression::createExpression(*child) );
        }
        // set inital value in accumlater ... can be done in more cleaner way
        initalizeAccumlator();
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        if (children.size() != 1) 
            throw std::runtime_error("AGG operator requires exactly 1 children");
        

        cudaStream_t& stream = arg.batchs[0]->stream; // 
        size_t batchSize = arg.batchs[0]->batchSize; // Assume ExpressionArg provides batchSize
        batchSizes.push_back(batchSize);

        ExpressionResult src = children[0]->evaluate(arg);
        
        char* nullset = src.nullset->bitset;
        void* src_data = src.result;

        // Allocate result memory
        
        switch (agg_type) {
        case AggType::MIN:
        switch (returnType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, YallaSQL::Kernel::MinOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset, batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_reduction_operators<float, YallaSQL::Kernel::MinOperator<float>>(
                    static_cast<float*>(src_data), static_cast<float*>(accumlator), nullset, batchSize, stream, initial_value_float);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, YallaSQL::Kernel::MinOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        case AggType::MAX:
        switch (returnType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, YallaSQL::Kernel::MaxOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset,  batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_reduction_operators<float, YallaSQL::Kernel::MaxOperator<float>>(
                    static_cast<float*>(src_data), static_cast<float*>(accumlator), nullset, batchSize, stream, initial_value_float);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, YallaSQL::Kernel::MaxOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        case AggType::AVG:
        case AggType::SUM:
        switch (returnType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, YallaSQL::Kernel::SumOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset, batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_sum_double_precision(
                    static_cast<float*>(src_data), static_cast<double*>(accumlator), nullset, batchSize, stream, initial_value_double);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, YallaSQL::Kernel::SumOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        default:
            break;
        }
        
        if(children[0]->exprType != ExpressionType::BOUND_VALUE)
            CUDA_CHECK(cudaFreeAsync(src_data, stream));

        return ExpressionResult();
    }

    //!must syncronize before calling
    void* getAggregate() {
        // cast double to float
        if(!tempAcc && returnType == DataType::FLOAT &&  agg_type == AggType::SUM) {
            CUDA_CHECK( cudaMalloc(&tempAcc, sizeof(double)) );
            YallaSQL::Kernel::launch_convert_double_to_float_kernel(static_cast<double*>(accumlator), tempAcc);
            cudaStreamSynchronize(cudaStreamDefault);
            return tempAcc;
        }

        // if (agg_type == AggType::AVG) {
        //     int* d_batchSizes;
        //     int *sum; 
        //     initial_value_int = 0;
            
        //     CUDA_CHECK( cudaMalloc((void**)&d_batchSizes, sizeof(int)*batchSizes.size())  );
        //     CUDA_CHECK( cudaMemcpy(d_batchSizes, batchSizes.data(), sizeof(int)*batchSizes.size(), cudaMemcpyHostToDevice) );
        //     CUDA_CHECK( cudaMalloc((void**)&sum, sizeof(int))  );
        //     CUDA_CHECK( cudaMemcpy(sum, &initial_value_int, sizeof(int), cudaMemcpyHostToDevice) );

        //     cudaStream_t st = cudaStreamDefault;
        //     YallaSQL::Kernel::launch_reduction_operators<int, YallaSQL::Kernel::SumOperator<int>>(
        //         static_cast<int*>(d_batchSizes), static_cast<int*>(sum), (uint32_t)batchSizes.size(), st, initial_value_int
        //     );
            
        //     //TODO: complete here...
        //     CUDA_CHECK(cudaFree(d_batchSizes));
        //     CUDA_CHECK(cudaFree(sum));
        // }

        return accumlator;
    }
    
    ~AggregateExpression() {
        if(tempAcc) { // keep only one value
            CUDA_CHECK( cudaFree(accumlator) );
        }
    }

private:
    void initalizeAccumlator() {
        const int bytes = returnType == DataType::FLOAT ? sizeof(double) :  getDataTypeNumBytes(returnType);
        CUDA_CHECK( cudaMalloc(&accumlator, bytes) );

        
        switch (agg_type) {
        case AggType::MIN:
        switch (returnType) {
            case DataType::INT:
                initial_value_int = std::numeric_limits<int>::max();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_int, bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::FLOAT:
                initial_value_float = std::numeric_limits<float>::max();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_float , bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::DATETIME:
                initial_value_date = std::numeric_limits<int64_t>::max();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_date , bytes, cudaMemcpyHostToDevice) );
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;

        case AggType::MAX:
        switch (returnType) {
            case DataType::INT:
                initial_value_int = std::numeric_limits<int>::min();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_int, bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::FLOAT:
                initial_value_float = std::numeric_limits<float>::min();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_float , bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::DATETIME:
                initial_value_date = std::numeric_limits<int64_t>::min();
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_date , bytes, cudaMemcpyHostToDevice) );
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;

        case AggType::AVG:
        case AggType::SUM:
        switch (returnType) {
            case DataType::INT:
                initial_value_int = 0;
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_int, bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::FLOAT:
                initial_value_double = 0;
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_float , bytes, cudaMemcpyHostToDevice) );
                break;
            case DataType::DATETIME:
                initial_value_date = 0;
                CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_date , bytes, cudaMemcpyHostToDevice) );
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;

        default:
            break;
        }

        CUDA_CHECK( cudaStreamSynchronize(cudaStreamDefault) );

            
    }


};
}