#pragma once
#include <memory>
#include "utils/macros.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "enums/data_type_utils.hpp"
#include "enums/device.hpp"
#include "batch.hpp"
#include "batch.hpp"
// #include ""
namespace our {


    enum class ExpressionType : __uint8_t {
        BOUND_REF,  // bound column in children 
        BOUND_FUNC, // +, -, * ....
        BOUND_VALUE,// bound constant value (2, 4, 100)

        CAST,
       
        COMPARISON,
        CONJUNCTION,
        NOT
        
    };


    struct ExpressionResult {
        // for async values +, - ....
        void* result;
        // num of return values
        size_t batchSize;
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
        bool required_op_child = true; // not like constant value etc..
        bool is_scalar = false;
        ExpressionType exprType;


        Expression(duckdb::Expression &expr) {
            alias = expr.alias;

            returnType = getDataTypeFromDuck( expr.return_type.id() );

            column = std::shared_ptr<Column>(new Column(alias, returnType));

        }
        Expression() {}
        // evaluate expression
        virtual ExpressionResult evaluate(ExpressionArg& arg) = 0;


        static std::unique_ptr<Expression> createExpression(duckdb::Expression &expr) ;

        virtual ~Expression() {}
    };


// aggregate expression Max, Min, Sum, Avg, .... //internal state update inside it
// binary expression +, -, *, /, % ... // take two batchs or one with scalar or two scalar and return a value
// unary expression -x, !x, ... // take one batch or one scalar and return a value
// constant expression 1, 2, 3, "hello", ... // take no batch and return a value
// bound_ref expression // take one batch and return a batch

}