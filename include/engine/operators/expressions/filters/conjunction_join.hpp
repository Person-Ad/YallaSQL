#pragma once
#include "engine/operators/expressions/expression.hpp"
#include "kernels/comparison_operators_kernel.hpp"

#include <duckdb/planner/expression/list.hpp>

namespace our {

    enum class ConjunctionJoinType : __uint8_t {
        AND, OR
    };
    [[nodiscard]] inline ConjunctionJoinType getConjunctionJoinType(duckdb::Expression &expr) {
        switch(expr.type) {
        case duckdb::ExpressionType::CONJUNCTION_AND:
            return ConjunctionJoinType::AND;  
        case duckdb::ExpressionType::CONJUNCTION_OR:
            return ConjunctionJoinType::OR;  
        default:
            throw std::runtime_error("Expression Type Not Supported: " + expr.ToString());
        }
    }  

    [[nodiscard]] inline ConjunctionJoinType getConjunctionJoinType(duckdb::ExpressionType type) {
        switch(type) {
        case duckdb::ExpressionType::CONJUNCTION_AND:
            return ConjunctionJoinType::AND;  
        case duckdb::ExpressionType::CONJUNCTION_OR:
            return ConjunctionJoinType::OR;  
        default:
            throw std::runtime_error("Expression Type Not Supported in Conjunction Join: ");
        }
    }  

class ConjunctionJoinExpression: public Expression {
    bool isneg, isjoin;
public:
    std::vector<std::unique_ptr<Expression>> children;
    ConjunctionJoinType conjunction_type;

    ConjunctionJoinExpression(duckdb::Expression &expr, bool isneg = false): Expression(expr), isneg(isneg) {
        exprType = ExpressionType::COMPARISON;
        conjunction_type = getConjunctionJoinType(expr);

        isjoin = false;
        // get left & right child
        auto& castExpr = expr.Cast<duckdb::BoundConjunctionExpression>();

        children.reserve(castExpr.children.size());
        for(auto& child: castExpr.children) {
            children.push_back( Expression::createExpression(*child) );
        }
    } 
    ConjunctionJoinExpression(duckdb::JoinCondition& cond): isneg(false) {
        exprType = ExpressionType::COMPARISON;
        conjunction_type = getConjunctionJoinType(cond.comparison);

        isjoin = true;
        
        children.reserve(2);
        children.push_back(Expression::createExpression(*cond.left));
        children.push_back(Expression::createExpression(*cond.right));

        children[0]->table_idx = 0;
        children[1]->table_idx = 1;
    } 
    // if there's multiple individual expressions in filter -> and them
    ConjunctionJoinExpression(std::unique_ptr<Expression> left, std::unique_ptr<Expression> right, bool isjoin = true): isneg(false), isjoin(isjoin) {
        exprType = ExpressionType::COMPARISON;
        conjunction_type = ConjunctionJoinType::AND;

        children.reserve(2);
        children.push_back(std::move(left));
        children.push_back(std::move(right));

        children[0]->table_idx = 0;
        children[1]->table_idx = 1;
    }

    ExpressionResult evaluate(ExpressionArg& arg) {
        ExpressionResult result;
        
        cudaStream_t& stream = arg.stream;
        
        ExpressionResult res_lhs = children[0]->evaluate(arg);
        ExpressionResult res_rhs = children[1]->evaluate(arg);
        
        int *actualOutSize;
        CUDA_CHECK(cudaMallocAsync((void**)&actualOutSize, sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(actualOutSize, 0, sizeof(int), stream));

        uint32_t* lhs = static_cast<uint32_t*>(res_lhs.result);
        uint32_t* rhs = static_cast<uint32_t*>(res_rhs.result);
  
        uint32_t actualLhsBs, actualRhsBs; 
        CUDA_CHECK(cudaMemcpyAsync(&actualLhsBs, res_lhs.d_batchSize, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream   ));
        CUDA_CHECK(cudaMemcpyAsync(&actualRhsBs, res_rhs.d_batchSize, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream   ));

        // Allocate result memory
        uint32_t cross_product_size = std::max(res_lhs.batchSize, res_rhs.batchSize);
        cross_product_size = CEIL_DIV( cross_product_size, 32 ); // store it as bitset
        
        uint32_t*  res;
        uint32_t*  mask;
        CUDA_CHECK(cudaMallocAsync(&res, 2*YallaSQL::MAX_ROWS_OUT_JOIN_OP*sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMallocAsync(&mask, cross_product_size * sizeof(uint32_t), stream));
        CUDA_CHECK(cudaMemset(mask, 0, cross_product_size * sizeof(uint32_t)));


        switch (conjunction_type) {
        case ConjunctionJoinType::AND:
            YallaSQL::Kernel::launch_and_join_operators(lhs, rhs, res, mask, actualOutSize, actualLhsBs, actualRhsBs, stream);
            break;
        case ConjunctionJoinType::OR:
            throw std::runtime_error("No OR in joins implemented!!\n");
            // YallaSQL::Kernel::launch_conditional_operators<bool, YallaSQL::Kernel::OROperator>(rhs, lhs, t_rhs, t_lhs, res, batchSize, stream, isneg);
            break;

        }


        CUDA_CHECK(cudaFreeAsync(lhs, stream));
        CUDA_CHECK(cudaFreeAsync(rhs, stream));

        result.result = res; 
        //! need ta access using actual block size
        result.d_batchSize = actualOutSize;
        result.batchSize = 0;
        //! danger
        // result.nullset = (children[0]->exprType != ExpressionType::BOUND_VALUE) ? res_lhs.nullset : res_rhs.nullset;
        return result;
    }

};
}