#pragma once
// #include <ctime>
#include <future>
#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <mutex>
#include <cstdio>
#include "config.hpp"
#include "db/table.hpp"
#include "enums/data_type_utils.hpp"
#include "enums/device.hpp"
#include "utils/macros.hpp"
#include "null_bit_set.hpp"

#define uint_32 unsigned int



class Batch {
    // must access it using getColumn, getItem
public:
    std::vector<void*> columnData;
    std::vector<std::shared_ptr<NullBitSet>> nullset;
    // the actual data that will be move between cpu & gpu
    // void* data = nullptr;
    // batch metadata
    Device location;
    uint_32 numCols;
    size_t batchSize; // rows
    size_t totalBytes;
    time_t lastAccessed;
    std::string filePath;
    cudaStream_t stream;

    // columns returned
    std::vector<std::shared_ptr<Column>> columns;
    // Protects access to `data` to not access it while movingp
    mutable std::mutex dataMutex;  

public:
    Batch(const std::vector<void*>& columnData, Device location, uint_32 batchSize, std::vector<std::shared_ptr<Column>>& columns, std::vector<std::shared_ptr<NullBitSet>>& nullset, cudaStream_t st = nullptr)
        : columnData(columnData), location(location), batchSize(batchSize), columns(columns), nullset(nullset) {
        // Create steam
        if(st) 
            stream = st;
        else 
            CUDA_CHECK(cudaStreamCreate(&stream));
        // Calculate total bytes for all column
        totalBytes = 0;
        for (size_t i = 0; i < columns.size(); ++i) {
            totalBytes += columns[i]->bytes * batchSize;
        }
        
        lastAccessed = time(nullptr); // update last accessed time
        numCols = columnData.size();
    }


  
    // will be used for projections
    void updateColumns(const std::vector<void*>& columnData, std::vector<std::shared_ptr<Column>> columns) {
        this->columnData = columnData;
        this->columns = columns;

        totalBytes = 0;
        for (size_t i = 0; i < columns.size(); ++i) {
            totalBytes += columns[i]->bytes * batchSize;
        }

        lastAccessed = time(nullptr); // update last accessed time
        numCols = columnData.size();
    }

    ~Batch() {
        std::unique_lock<std::mutex> lock(dataMutex);
        
        if(location == Device::FS) 
            std::remove(filePath.c_str());

        for (void* ptr : columnData) {
            if (ptr) {
                if (location == Device::CPU) {
                    CUDA_CHECK( cudaFreeHost(ptr) );
                } else if(location == Device::GPU) {
                    CUDA_CHECK( cudaFreeAsync(ptr, stream) );
                }
            }
        }

        // cudaStreamDestroy(stream);
        
    }
    // Move constructor
    // Batch(Batch&&) = default;
    // Batch& operator=(Batch&&) = default;

    // // Delete copy to prevent unintended copies
    // Batch(const Batch&) = delete;
    // Batch& operator=(const Batch&) = delete;

    // Getters
    void* getColumn(uint_32 colIndex) const {
        std::unique_lock<std::mutex> lock(dataMutex);
        
        // lastAccessed = time(nullptr);
        return columnData[colIndex];
    }
    template <typename T = std::string>
    T* getItem(uint_32 colIndex, uint_32 rowIndex) const {
        std::unique_lock<std::mutex> lock(dataMutex);

        // lastAccessed = time(nullptr);
        char* colPtr = static_cast<char*>(columnData[colIndex]);
        uint32_t typeSize = columns[colIndex]->bytes;
        return reinterpret_cast<T*>(colPtr + rowIndex * typeSize);
    }
    //! Take care it only happen in CPU||GPU
    void removeColumn(uint_32 colIndex);

    void moveTo(Device target);

    void print();
private:
    // downward 
    void moveGpuToCpu() ;
    void moveCpuToFs() ;
    // upward
    void moveFsToCpu() ;
    void moveCpuToGpu() ;

    std::future<void> serializeBatchAsync(const Batch& batch, const std::string& filePath, std::function<void(bool)> callback = nullptr);

    std::future<bool> deserializeBatchAsync(Batch& batch);
};
