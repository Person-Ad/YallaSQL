#pragma once

#include "db/db.hpp"
#include "db/table.hpp"
#include "csv-parser/csv.hpp"
#include "engine/operators/operator.hpp"

namespace YallaSQL {

class GetOperator final : public Operator {

private: 
    // instance of db
    DB* db = nullptr;
    // table I work on
   const Table* table = nullptr;
    // represent state of operator    
    csv::CSVReader *reader = nullptr;
    uint32_t currRow = 0;
    std::vector<char*> buffer; // reserve bytes by columnIdx
    std::vector<std::string> csvNames; 

public:
    // inherit from operator
    using Operator::Operator; 
    // GetOperator(const duckdb::LogicalOperator& op): Operator(op) {}
    // opening csv file of table memory map & set schema of return
    void init() override;
    // return next batch index or 0 if empty
    BatchID next(CacheManager& cacheManager) override;
    // delete buffer & reader
    ~GetOperator() override;

private:
    // store buffer into new pointer to pass it
    std::unique_ptr<Batch> storeBuffer(uint32_t batchSize);
};
} // YallaSQL