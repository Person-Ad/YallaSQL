#include "engine/operators/get_operator.hpp"
#include "engine/operators/operator.hpp"
#include "kernels/string_kernel.hpp"
#include "utils/macros.hpp"
#include "db/table.hpp"
#include "logger.hpp"
#include "config.hpp"
#include "db/db.hpp"

#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

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
    std::vector<DataType> columnsType; // for calculating optimal batch size
    auto &columnIndexs = logicalCastOp.GetColumnIds();
    columns.reserve(columnIndexs.size());
    csvNames.reserve(columnIndexs.size());
    for(auto& columnIndex : columnIndexs) {
        auto idxInTable = columnIndex.GetPrimaryIndex();
        if(idxInTable > table->columnsOrdered.size()) {
            idxInTable = 0;
        }
        columns.push_back(table->columnsOrdered[idxInTable]);
        csvNames.push_back(table->csvNameColumn[idxInTable]);
        columnsType.push_back(table->columnsType[idxInTable]);
    }
    // === set metadata ===
    batchSize = calculateOptimalBatchSize(columnsType);
    LOG_INFO(getLogger(""), "Base Batch Size {}", batchSize);

    // ==== reserve buffer memory ==== 
    buffer.resize(columns.size()); // reserve buffer foreach column
    for(uint i = 0; i < columns.size(); i++) {
        // buffer[i] =  new char[ columns[i]->bytes * batchSize ];  // num of bytes need of field with data type of column * num of fields i have
        CUDA_CHECK(cudaMallocHost((void**)&buffer[i], columns[i]->bytes * batchSize));
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
    // -- iterator loop --
    csv::CSVRow row;
    std::vector<std::string> row_data(table->columns.size());
    std::vector<std::vector<char>> h_nullset(table->columns.size()); //[col][row]

    while(rowIndex < batchSize && reader->read_row(row)) {
       // Immediately copy ALL row data to local storage
        for (size_t i = 0; i < csvNames.size(); i++) {
            row_data[i] = row[csvNames[i]].get<std::string>();
        }
        // store in buffer
        colIndex = 0;
        for(std::shared_ptr<Column> column: columns) {
            const std::string& valueStr = row_data[colIndex];
            h_nullset[colIndex].push_back(valueStr.empty());

            if(column->type == DataType::INT) {
                int value = valueStr.empty() ? 0 : std::stoi(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else if(column->type == DataType::FLOAT) {
                float value = valueStr.empty() ? 0 : std::stof(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else if(column->type == DataType::DATETIME) {
                int64_t value = valueStr.empty() ? 0 : getDateTime(valueStr);
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes);
            } else { //string
                YallaSQL::Kernel::String value;
                value.set(valueStr.c_str());
                std::memcpy(buffer[colIndex] + rowIndex * column->bytes, &value, column->bytes); // 1 for \0
            }
            colIndex++;
        }

        rowIndex++;
    }
    if (rowIndex == 0) {
        isFinished = true;
        return 0;
    }
    // --- update state ---
    std::unique_ptr<Batch> batch = storeBuffer(rowIndex, h_nullset);
    // batch->moveTo(Device::GPU);
    auto batchId = cacheManager.putBatch(std::move(batch));
    currRow += rowIndex;

    return batchId;
}
// delete buffer & reader
GetOperator::~GetOperator() {
    if(reader) delete reader;
    // if(buffer) {
        // First free each array element
        for(uint i = 0; i < columns.size(); i++) {
            // delete[] buffer[i];
            CUDA_CHECK(cudaFreeHost(buffer[i]));
        }
        // Then free the array of pointers
        // delete[] buffer;
    // }
}


std::unique_ptr<Batch> GetOperator::storeBuffer(uint32_t batchSize, std::vector<std::vector<char>>& h_nullset) {
    // need to store in column wise
    uint32_t stride = 0;
    uint32_t colIndex = 0;
    
    std::vector<void*> data(columns.size());
    std::vector<std::shared_ptr<NullBitSet>> nullset(columns.size());

    for(std::shared_ptr<Column> column: columns) {
        unsigned int colSize = column->bytes * batchSize;
        // allocate memory for column
        CUDA_CHECK( cudaMalloc(&data[colIndex], colSize) );
        CUDA_CHECK( cudaMemcpy(data[colIndex], buffer[colIndex], colSize, cudaMemcpyHostToDevice) );
        // Init & Move Nullset to device
        nullset[colIndex] = std::shared_ptr<NullBitSet>( new NullBitSet(h_nullset[colIndex].data(), h_nullset[colIndex].size())  );

        stride += colSize;
        ++colIndex;
    }
    // I hate myself alot
    cudaStreamSynchronize(cudaStreamDefault);

    return std::unique_ptr<Batch>(new Batch(data, Device::GPU, batchSize, columns, nullset));
}
}