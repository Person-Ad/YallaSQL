#include "engine/operators/get_operator.hpp"
#include "engine/operators/operator.hpp"
#include "utils/macros.hpp"
#include "db/table.hpp"
#include "config.hpp"
#include "db/db.hpp"

#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/execution/operator/scan/physical_column_data_scan.hpp>
#include <duckdb/function/table/read_csv.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/common/types/batched_data_collection.hpp>
#include <duckdb/common/file_system.hpp>
// #include <duckdb/function/t>


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
    // ==== set columns ====
    //! Shuffle As Physical Duckdb Expect :<
    auto &columnIndexs = logicalCastOp.GetColumnIds();
    columns.reserve(columnIndexs.size());
    csvNames.reserve(columnIndexs.size());
    for(auto& columnIndex : columnIndexs) {
        auto idxInTable = columnIndex.GetPrimaryIndex();
        columns.push_back(table->columnsOrdered[idxInTable]);
        csvNames.push_back(table->csvNameColumn[idxInTable]);
    }
    // === set metadata ===
    batchSize = calculateOptimalBatchSize(table->columnsType);
    // ==== reserve buffer memory ==== 
    buffer = new char*[columns.size()]; // reserve buffer foreach column
    for(uint i = 0; i < columns.size(); i++) {
        buffer[i] = new char[ columns[i]->bytes *batchSize ];  // num of bytes need of field with data type of column * num of fields i have
    }
    // ==== change state ====
    isInitalized = true;
}


BatchID GetOperator::next(CacheManager& cacheManager) {
    if(!isInitalized) init();
    if(isFinished) return 0;
    // if(reader->())
    // -- temp parameters to not miss with curr state if error happen --
    uint32_t colIndex = 0;
    uint32_t rowIndex = 0; // equivalent to current batch size
    uint32_t iteratorRow = currRow;
    // -- iterator loop --
    csv::CSVRow row;
    std::vector<std::string> row_data(table->columns.size());
    // YallaSQL::UTILS::MEAS
    while(rowIndex < batchSize && reader->read_row(row)) {
        // Copy row fields to avoid use-after-free
        colIndex = 0;
        for (const auto& column : columns) {
            row_data[colIndex] = row[csvNames[colIndex]].get<std::string>();
            colIndex++;
        }
        // store in buffer
        colIndex = 0;
        for(std::shared_ptr<Column> column: columns) {
            const std::string& valueStr = row_data[colIndex];
            if(column->type == DataType::INT) {
                int value = std::stoi(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else if(column->type == DataType::FLOAT) {
                float value = std::stof(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else if(column->type == DataType::DATETIME) {
                int64_t value = getDateTime(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else { //string
                std::string value = valueStr.substr(0, column->bytes - 1);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, value.c_str(), value.size() + 1); // 1 for \0
            }
            colIndex++;
        }

        iteratorRow++;
        rowIndex++;
    }
    if (rowIndex == 0) {
        isFinished = true;
        return 0;
    }
    // --- update state ---
    std::unique_ptr<Batch> batch = storeBuffer(rowIndex);
    // batch->moveTo(Device::GPU);
    auto batchId = cacheManager.putBatch(std::move(batch));
    currRow += rowIndex;

    return batchId;
}
// delete buffer & reader
GetOperator::~GetOperator() {
    if(reader) delete reader;
    if(buffer) {
        // First free each array element
        for(uint i = 0; i < columns.size(); i++) {
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
    
    std::vector<void*> data(columns.size());

    for(std::shared_ptr<Column> column: columns) {
        unsigned int colSize = column->bytes * batchSize;
        // allocate memory for column
        CUDA_CHECK( cudaMallocHost(&data[colIndex], colSize) );
        std::memcpy(data[colIndex], buffer[colIndex], colSize);
        // update 
        stride += colSize;
        ++colIndex;
    }



    return std::unique_ptr<Batch>(new Batch(data, Device::CPU, batchSize, columns));
}

void GetOperator::readBatchFaster() {
    // auto& context = db->duckdb().context;
    // // set options
    // duckdb::DialectOptions dialect_options;    // Assuming CSV has header
    // dialect_options.header = true;
    // dialect_options.num_cols = table->numCols;
    // dialect_options.skip_rows = 1;
    //
    // duckdb::CSVReaderOptions options;
    // options.dialect_options = dialect_options;
    // options.file_path = table->path;
    // options.auto_detect = false;  // We know our schema
    // // Set up column mapping based on our table schema
    // for (auto& column : table->columnsOrdered) {
    //     LogicalType col_type;
    //     switch (column->type) {
    //         case DataType::INT:
    //             col_type = duckdb::LogicalType::INTEGER;
    //             break;
    //         case DataType::FLOAT:
    //             col_type = duckdb::LogicalType::FLOAT;
    //             break;
    //         case DataType::DATETIME:
    //             col_type = duckdb::LogicalType::TIMESTAMP;
    //             break;
    //         default: // STRING
    //             col_type = duckdb::LogicalType::VARCHAR;
    //             break;
    //     }
    //
    //     options.sql_type_list.push_back(col_type);
    //
    // }
    //
    // // Create CSV reader and scanner
    // auto &fs = duckdb::FileSystem::GetFileSystem(*context);
    // auto csv_reader = duckdb::ReadCSV::OpenCSV(context, options, fs);



}


}