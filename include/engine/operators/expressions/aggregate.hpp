#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/reduction_kernel.hpp"
#include "kernels/binary_operators_kernel.hpp"
namespace our {

    enum class AggType : __uint8_t {
        MAX, MIN, SUM, AVG, COUNT, COUNT_STAR
    };

    [[nodiscard]] inline AggType getAggType(std::string str) {
        if(str == "max") return AggType::MAX;
        if(str == "min") return AggType::MIN;
        if(str == "sum") return AggType::SUM;
        if(str == "count") return AggType::COUNT;
        if(str == "count_star") return AggType::COUNT_STAR;
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
    int* counter;// counter for average

    DataType srcType;
    int total_rows = 0;
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
        if(children.size() > 0) {
            srcType = children[0]->returnType;
        }
        // set inital value in accumlater ... can be done in more cleaner way
        initalizeAccumlator();
    } 

    ExpressionResult evaluate(ExpressionArg& arg) {
        if(agg_type == AggType::COUNT_STAR) {
            total_rows += arg.batchs[0]->batchSize;
            return ExpressionResult();
        }
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
        case AggType::COUNT_STAR:
            total_rows += batchSize; 
            break; 
        case AggType::MIN:
        switch (srcType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, int, YallaSQL::Kernel::MinOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset, batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_reduction_operators<float, float, YallaSQL::Kernel::MinOperator<float>>(
                    static_cast<float*>(src_data), static_cast<float*>(accumlator), nullset, batchSize, stream, initial_value_float);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, int64_t, YallaSQL::Kernel::MinOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        case AggType::MAX:
        switch (srcType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, int, YallaSQL::Kernel::MaxOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset,  batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_reduction_operators<float, float, YallaSQL::Kernel::MaxOperator<float>>(
                    static_cast<float*>(src_data), static_cast<float*>(accumlator), nullset, batchSize, stream, initial_value_float);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, int64_t, YallaSQL::Kernel::MaxOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        case AggType::COUNT:
            YallaSQL::Kernel::launch_reduction_count_notnull(nullset, static_cast<int*>(accumlator), nullset, batchSize, stream, initial_value_int);
            break;
        case AggType::AVG:
            YallaSQL::Kernel::launch_reduction_count_notnull(nullset, counter, nullset, batchSize, stream, initial_value_int);
        case AggType::SUM:
        switch (srcType) {
            case DataType::INT:
                YallaSQL::Kernel::launch_reduction_operators<int, int, YallaSQL::Kernel::SumOperator<int>>(
                    static_cast<int*>(src_data), static_cast<int*>(accumlator), nullset, batchSize, stream, initial_value_int);
                break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_reduction_operators<float, double, YallaSQL::Kernel::SumOperator<double>>(
                    static_cast<float*>(src_data), static_cast<double*>(accumlator), nullset, batchSize, stream, initial_value_double);
                break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_reduction_operators<int64_t, int64_t, YallaSQL::Kernel::SumOperator<int64_t>>(
                    static_cast<int64_t*>(src_data), static_cast<int64_t*>(accumlator), nullset, batchSize, stream, initial_value_date);
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        break;
        default:
            break;
        }
        
        if(agg_type != AggType::COUNT_STAR &&  children[0]->exprType != ExpressionType::BOUND_VALUE)
            CUDA_CHECK(cudaFreeAsync(src_data, stream));

        return ExpressionResult();
    }

    //!must syncronize before calling
    void* getAggregate() {
        cudaStream_t stream = cudaStreamDefault;
        // cast double to float
        if(agg_type == AggType::SUM && srcType == DataType::FLOAT) {
            if(tempAcc) return tempAcc;

            CUDA_CHECK( cudaMalloc(&tempAcc, sizeof(float)) );
            YallaSQL::Kernel::launch_convert_double_to_float_kernel(static_cast<double*>(accumlator), tempAcc);
            cudaStreamSynchronize(stream);
            return tempAcc;
        }

        if (agg_type == AggType::AVG) {
            if(tempAcc) return tempAcc;
            //
            CUDA_CHECK( cudaMalloc(&tempAcc, sizeof(float)) );
            switch (srcType){
            case DataType::INT:
                YallaSQL::Kernel::launch_div_avg<int>(counter, static_cast<int*>(accumlator), tempAcc);
            break;
            case DataType::FLOAT:
                YallaSQL::Kernel::launch_div_avg<double>(counter, static_cast<double*>(accumlator), tempAcc);
            break;
            case DataType::DATETIME:
                YallaSQL::Kernel::launch_div_avg<int64_t>(counter, static_cast<int64_t*>(accumlator), tempAcc);
            break;
            default:
                break;
            }
            cudaStreamSynchronize(stream);

            return tempAcc;
        }

        if(agg_type == AggType::COUNT_STAR) {
            if(tempAcc) return tempAcc;  
            CUDA_CHECK( cudaMalloc(&tempAcc, sizeof(int)) );
            CUDA_CHECK( cudaMemcpy(tempAcc, &total_rows, sizeof(int), cudaMemcpyHostToDevice) );
            cudaStreamSynchronize(stream);
            return tempAcc;
        }

        return accumlator;
    }
    
    ~AggregateExpression() {
        if(tempAcc) { // keep only one value
            CUDA_CHECK( cudaFree(accumlator) );
        }
    }

private:
    void initalizeAccumlator() {
        // allocate double in this case only for stability division
        const int bytes = (srcType == DataType::FLOAT && (agg_type == AggType::SUM || agg_type == AggType::AVG)) ? 
                            sizeof(double) :  getDataTypeNumBytes(srcType);
        CUDA_CHECK( cudaMalloc(&accumlator, bytes) );

        
        switch (agg_type) {
        case AggType::MIN:
        switch (srcType) {
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
        switch (srcType) {
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

        case AggType::COUNT:
            initial_value_int = 0;
            CUDA_CHECK( cudaMemcpy(accumlator, &initial_value_int, bytes, cudaMemcpyHostToDevice) );
            break;
        case AggType::AVG:
            initial_value_int = 0;
            CUDA_CHECK( cudaMalloc(&counter, sizeof(int)) );
            CUDA_CHECK( cudaMemcpy(counter, &initial_value_int, sizeof(int), cudaMemcpyHostToDevice) );
        case AggType::SUM:
        switch (srcType) {
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