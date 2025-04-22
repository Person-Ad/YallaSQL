#include "engine/operators/get_operator.hpp"
#include "engine/operators/operator.hpp"
#include "utils/macros.hpp"
#include "db/table.hpp"
#include "config.hpp"
#include "db/db.hpp"

#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include "csv-parser/csv.hpp"

namespace YallaSQL {

// opening csv file of table memory map & set schema of return
void GetOperator::init() {
    if(isInitalized) return; // don't initalize again
    // initiate db
    db = DB::getInstance();

    // get table we get result from
    const LogicalGet& logicalCastOp = logicalOp.Cast<LogicalGet>();
    const std::string& tableName = logicalCastOp.GetTable().get()->name;
    table = db->getTable(tableName);
    // open reader of file
    reader = new csv::CSVReader(table->path);
    // ==== set metadata ==== 
    columns = table->columnsOrdered;
    batchSize = calculateOptimalBatchSize(table->columnsType);
    // ==== reserve buffer memory ==== 
    buffer = new char*[table->numCols]; // reserve buffer foreach column
    for(uint i = 0; i < table->numCols; i++) {
        buffer[i] = new char[
                            table->columnsOrdered[i]->bytes *  // num of bytes need of field with data type of column
                            batchSize]; // num of fields i have
    }
    // ==== change state ====
    isInitalized = true;
}


BatchID GetOperator::next(CacheManager& cacheManager) {
    if(!isInitalized) init();
    if(isFinished) return 0;
    if (reader->eof()) { isFinished = 1; return 0; }
    // if(reader->())
    // -- temp parameters to not miss with curr state if error happen --
    uint32_t colIndex = 0;
    uint32_t rowIndex = 0; // equivalent to current batch size
    uint32_t iteratorRow = currRow;
    // -- iterator loop --
    csv::CSVRow row;
    bool x = false;
    MEASURE_EXECUTION_TIME_MICRO_LOGGER(logger, "Reading Row", 
        x = rowIndex < batchSize && reader->read_row(row);
    )
    // YallaSQL::UTILS::MEAS
    MEASURE_EXECUTION_TIME_LOGGER(logger, "Reading A Batch",

    while(rowIndex < batchSize && reader->read_row(row)) {
        colIndex = 0;
        for(std::shared_ptr<Column> column: table->columnsOrdered) {
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
                
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, value.c_str(), value.size() + 1); // 1 for \0
            }
            colIndex++;
        }

        iteratorRow++;
        rowIndex++;
    }
        )


    // --- update state ---

    std::unique_ptr<Batch> batch = storeBuffer(rowIndex);
    auto batchId = cacheManager.putBatch(std::move(batch));
    currRow += rowIndex;

    return batchId;
}
// delete buffer & reader
GetOperator::~GetOperator() {
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


std::unique_ptr<Batch> GetOperator::storeBuffer(uint32_t batchSize) {
    // need to store in column wise
    uint32_t stride = 0;
    uint32_t colIndex = 0;
    
    std::vector<void*> data(table->numCols);

    std::vector<std::shared_ptr<Column>> columns;
    for(std::shared_ptr<Column> column: table->columnsOrdered) {
        unsigned int colSize = column->bytes * batchSize;
        // allocate memory for column
        CUDA_CHECK( cudaMallocHost(&data[colIndex], colSize) );
        std::memcpy(data[colIndex], buffer[colIndex], colSize);
        // update 
        stride += colSize;
        ++colIndex;
        columns.push_back(column);
    }

    return std::unique_ptr<Batch> (new Batch(data, Device::CPU, batchSize, columns));
}

}