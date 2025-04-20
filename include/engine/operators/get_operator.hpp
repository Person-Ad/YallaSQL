#pragma once

#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include "db/db.hpp"
#include "config.hpp"
#include "db/table.hpp"
#include "csv-parser/csv.hpp"
#include "engine/operators/operator.hpp"

namespace YallaSQL {

class GetOperator: Operator {

private: 
    // instance of db
    DB* db = nullptr;
    // table I work on
    const Table* table = nullptr;
    // columns I return 
    std::vector<std::string> columnNames;
    // schema I return
    std::vector<DataType> schema;
    // best batchsize of current table
    unsigned int batchSize = 0;
    // represent state of operator    
    csv::CSVReader *reader = nullptr;
    uint32_t currRow = 0;
    char** buffer = nullptr; // reserve bytes by columnIdx

public:
    // opening csv file of table memory map & set schema of return
    void init(const duckdb::LogicalOperator& logicalOp) {
        // initiate db
        db = DB::getInstance();

        // get table we get result from
        const LogicalGet& logicalCastOp = logicalOp.Cast<LogicalGet>();
        const std::string& tableName = logicalCastOp.GetTable().get()->name;
        
        table = db->getTable(tableName);
        
        reader = new csv::CSVReader(table->path);

        batchSize = calculateOptimalBatchSize(table->columnsType);
        // reserve buffer memory 
        buffer = new char*[table->numCols]; // reserve buffer foreach column
        for(uint i = 0; i < table->numCols; i++) {
            buffer[i] = new char[
                                table->columnsOrdered[i]->bytes *  // num of bytes need of field with data type of column
                                batchSize]; // num of fields i have
        }
    }
    // return next batch index or 0 if empty
    BatchID next(CacheManager& cacheManager) {
        // -- temp parameters to not miss with curr state if error happen --
        uint32_t colIndex = 0;
        uint32_t rowIndex = 0; // equivalent to current batch size
        uint32_t iteratorRow = currRow;
        // -- iterator loop --
        csv::CSVRow row;
        while(rowIndex < batchSize && reader->read_row(row)) {
            colIndex = 0;
            for(const auto* column: table->columnsOrdered) {
                if(column->type == DataType::INT) {
                    int value = row[column->name].get<int>();
                    std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
                } else if(column->type == DataType::FLOAT) {
                    float value = row[column->name].get<float>();
                    std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
                } else if(column->type == DataType::DATETIME) {
                    std::string valueStr = row[column->name].get<>();
                    int64_t value = getDateTime(valueStr);
                    std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
                } else { //string
                    std::string value = row[column->name].get<std::string>();
                                value = value.substr(0, column->bytes - 1);
                    
                    std::memcpy(buffer[colIndex] + rowIndex * column->bytes, value.c_str(), column->bytes); // 1 for \0
                }
                colIndex++;
            }

            iteratorRow++;
            rowIndex++;
        }


        // --- update state ---

        Batch* batch = storeBuffer(rowIndex);
        auto batchId = cacheManager.putBatch(batch);
        currRow += rowIndex;

        return batchId;
    }

    virtual ~GetOperator() {
        if(reader) delete reader;
        if(buffer) {
            // First free each array element
            for(uint i = 0; i < table->numCols; i++) {
                delete[] buffer[i];
            }
            // Then free the array of pointers
            delete[] buffer;
        }
    }

private:
    Batch* storeBuffer(uint32_t batchSize) {
        // need to store in column wise
        uint32_t stride = 0;
        uint32_t colIndex = 0;
        
        char* data = (char*)malloc(table->rowBytes * batchSize);

        for(const auto* column: table->columnsOrdered) {
            std::memcpy(data + stride, buffer[colIndex], column->bytes * batchSize);
            stride += column->bytes * batchSize;
            ++colIndex;
        }

        return new Batch((void *) data, Device::CPU, table->columnsType, batchSize);
    }

};
} // YallaSQL