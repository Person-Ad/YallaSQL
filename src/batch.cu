// #include <ctime>
#include <fstream>
#include <cuda_runtime.h>
#include "batch.hpp"
#include "utils/macros.hpp"


void Batch::moveTo(Device target) {
    lastAccessed = time(nullptr);
    if(location == target) return; // no need to move
    if(location == Device::CPU && target == Device::GPU) {
        moveCpuToGpu();
    } else if(location == Device::CPU && target == Device::FS) {
        moveCpuToFs();
    } else if(location == Device::GPU && target == Device::CPU) {
        moveGpuToCpu();
    } else if(location == Device::GPU && target == Device::FS) {
        moveGpuToCpu();
        moveCpuToFs();
    } else if(location == Device::FS && target == Device::CPU) {
        moveFsToCpu();
    } else if(location == Device::FS && target == Device::GPU) {
        moveFsToCpu();
        moveCpuToGpu();
    }
}

void Batch::moveGpuToCpu() {
    std::unique_lock<std::mutex> lock(dataMutex);

    std::vector<void*> cpuData(columnData.size());
    // move each column to CPU 
    for (size_t i = 0; i < columnData.size(); ++i) {
        uint32_t colBytes = columns[i]->bytes * batchSize;
        void* buffer = nullptr;
        // allocate CPU memory
        CUDA_CHECK( cudaMallocHost((void**)&buffer, colBytes) );

        // copy data from GPU to CPU
        cudaMemcpy(buffer, columnData[i], colBytes, cudaMemcpyDeviceToHost);
        // free old gpu memory
        cudaFree(columnData[i]);
        // store pointer
        cpuData[i] = buffer;
    }

    columnData = std::move(cpuData);
    location = Device::CPU;
}

void Batch::moveCpuToGpu() {
    std::unique_lock<std::mutex> lock(dataMutex);

    std::vector<void*> gpuData(columnData.size());
    // move each column to GPU
    for (size_t i = 0; i < columnData.size(); ++i) {
        uint32_t colBytes = columns[i]->bytes * batchSize;
        void* buffer = nullptr;
        // allocate GPU memory
        CUDA_CHECK( cudaMalloc((void**)&buffer, colBytes) );
        // copy data from CPU to GPU
        cudaMemcpy(buffer, columnData[i], colBytes, cudaMemcpyHostToDevice);
        // free old CPU memory
        cudaFreeHost(columnData[i]); 
        // store the GPU pointer
        gpuData[i] = buffer;
    }

    columnData = std::move(gpuData);
    location = Device::GPU;
}

void Batch::moveCpuToFs() {
    auto now = std::chrono::system_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::string path = YallaSQL::cacheDir + "/" + "batch_" + std::to_string(millis) + ".bin";

    dataMutex.lock(); // will be released inside callback

    serializeBatchAsync(*this, path, [this, path](bool success) {
        std::unique_lock<std::mutex> lock(this->dataMutex, std::adopt_lock);

        if (success) {
            for (void* ptr : this->columnData) {
                if (ptr) cudaFreeHost(ptr);
            }
            this->location = Device::FS;
            this->filePath = path;
        } else {
            std::cerr << "Error: Failed to serialize batch to file.\n";
        }
    });
}


// upward
void Batch::moveFsToCpu() {
    std::unique_lock<std::mutex> lock(dataMutex);

    bool success = deserializeBatchAsync(*this).get();
    if (!success) {
        throw std::runtime_error("Failed to deserialize batch from file.");
    }
    std::remove(filePath.c_str()); // remove the file after loading
    location = Device::CPU;
    filePath.clear(); // clear the file path
}

void Batch::removeColumn(uint_32 colIndex) {
    std::unique_lock<std::mutex> lock(dataMutex);
    
    lastAccessed = time(nullptr);
    // can't remove columns if batch in FS //TODO: could be optimized by store each column in file
    if(location == Device::FS) moveFsToCpu();
    if(location == Device::CPU) {
        cudaFreeHost(columnData[colIndex]);
    } 
    else {
        cudaFree(columnData[colIndex]);
    }
    numCols--;
    totalBytes -= columns[colIndex]->bytes * batchSize;
    columnData.erase(columnData.begin() + colIndex);
    columns.erase(columns.begin() + colIndex);
}

// #ifdef YALLASQL_DEBUG
void Batch::print() {
    moveTo(Device::CPU);
    uint_32 colIndex = 0;
    for(auto& column: this->columns) {
        std::cout << column->name << " ";
        if(column->type == DataType::STRING) std::cout << "(str): ";
        else if(column->type == DataType::INT) std::cout << "(int): ";
        else if(column->type == DataType::FLOAT) std::cout << "(float): ";
        else if(column->type == DataType::DATETIME) std::cout << "(datetime): ";

        for(uint_32 rowIndex = 0;rowIndex < batchSize;rowIndex++) {
            if(column->type == DataType::STRING) std::cout << getItem<const char>(colIndex, rowIndex);
            else if(column->type == DataType::INT) std::cout << *getItem<int>(colIndex, rowIndex);
            else if(column->type == DataType::FLOAT) std::cout << *getItem<float>(colIndex, rowIndex);
            else if(column->type == DataType::DATETIME) {
                std::cout << getDateTimeStr( *getItem<int64_t>(colIndex, rowIndex) );
            }

            std::cout << ", ";
        }


        std::cout << "\n";
        ++colIndex;
    }
}


std::future<void> Batch::serializeBatchAsync(const Batch& batch, const std::string& filePath, std::function<void(bool)> callback) {
    return std::async(std::launch::async, [&batch, filePath, callback]() {
        try {
            std::ofstream ofs(filePath, std::ios::binary);
            if (!ofs.is_open()) {
                if (callback) callback(false);
                return;
            }

            for (size_t i = 0; i < batch.columnData.size(); ++i) {
                uint32_t colBytes = batch.columns[i]->bytes * batch.batchSize;
                ofs.write(static_cast<const char*>(batch.columnData[i]), colBytes);
            }

            ofs.close();
            if (callback) callback(true);
        } catch (const std::exception& e) {
            if (callback) callback(false);
        }
    });
}

std::future<bool> Batch::deserializeBatchAsync(Batch& batch) {
    return std::async(std::launch::async, [&batch]() -> bool {
        try {
            std::ifstream ifs(batch.filePath, std::ios::binary);
            if (!ifs.is_open()) {
                return false;
            }

            for (size_t i = 0; i < batch.columns.size(); ++i) {
                uint32_t colBytes = batch.columns[i]->bytes * batch.batchSize;
                CUDA_CHECK(cudaMallocHost(&batch.columnData[i], colBytes));
                ifs.read(static_cast<char*>(batch.columnData[i]), colBytes);
            }

            ifs.close();
            return true;
            // Create Batch
            // return new Batch(data, location, batchSize, columns);
        } catch (const std::exception& e) {
            // Handle error (e.g., log it)
            std::cerr << "Error deserializing batch: " << e.what() << std::endl;
            return false;
        }
    });
}