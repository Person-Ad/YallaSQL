#pragma once
// #include <ctime>
#include <cuda_runtime.h>
#include <vector>
#include "config.hpp"
#include "enums/data_type.hpp"
#include "enums/device.hpp"

#define uint_32 unsigned int

// so the data store in **column store**
// to access 2nd column need to jump (N * bytes_datatype[0] + N * bytes_datatype[0])
//                                   = N (bytes_datatype_0 + bytes_datatype[1] ) # let's save it as prefix
// to access bytes_datatype from prefix sum for col_0 sum[1] - sum[0] 
//                                          for col_1 sum[2] - sum[1]
//                                          for col_m sum[m+1] - sum[m]
struct Batch {
    // the actual data that will be move between cpu & gpu
    void* data = nullptr;
    // cpu | gpu
    Device location;
    // number of rows / values per column
    uint_32 batchSize;
    // type foreach column 
    std::vector<DataType> types;
    // byte size of type foreach column 
    std::vector<uint_32> bytesSum; // it's not memory efficent may be remove it and handl logic somewhere else
                                        // size = no_of_columns + 1
    // it will be used for removeing data from gpu
    // time_t lastAccessed;

    Batch(void* data, Device location, std::vector<DataType> types, uint_32 batchSize)
        : data(data), location(location), batchSize(batchSize), types(types) {
        // num of cols = types.size()
        bytesSum = std::vector<uint_32>(types.size() + 1, 0);
        // generate prefix sum
        for(uint_32 i = 0;i < types.size(); i++) {
            bytesSum[i + 1] = bytesSum[i] + getDataTypeNumBytes(types[i]);
        }
    }

    ~Batch() {
        if(data && location == Device::CPU) 
            free(data);
        else if(data)
            cudaFree(data);
        
    }


    void *getColumn(uint_32 colIndex) {
        return static_cast<char*>(data) 
                + batchSize * (bytesSum[colIndex]); // jump to column
    }
    template <typename T = std::string>
    T *getItem(uint_32 colIndex, uint_32 rowIndex) {
        return (T*)(static_cast<char*>(data) 
                + batchSize * (bytesSum[colIndex]) // jump to column
                + rowIndex * (bytesSum[colIndex + 1] - bytesSum[colIndex])); // jump to row
    }

    void moveTo(Device target) {
        if(target == Device::CPU) {

        } else {

        }
    }


    // #ifdef YALLASQL_DEBUG
    void print() {
        uint_32 colIndex = 0;
        for(auto type: types) {
            if(type == DataType::STRING) std::cout << "Col (str): ";
            else if(type == DataType::INT) std::cout << "Col (int): ";
            else if(type == DataType::FLOAT) std::cout << "Col (float): ";
            else if(type == DataType::DATETIME) std::cout << "Col (datetime): ";

            for(uint_32 rowIndex = 0;rowIndex < batchSize;rowIndex++) {
                if(type == DataType::STRING) std::cout << getItem<const char>(colIndex, rowIndex);
                else if(type == DataType::INT) std::cout << *getItem<int>(colIndex, rowIndex);
                else if(type == DataType::FLOAT) std::cout << *getItem<float>(colIndex, rowIndex);
                else if(type == DataType::DATETIME) {
                    std::cout << getDateTimeStr( *getItem<int64_t>(colIndex, rowIndex) );
                }

                std::cout << ", ";
            }


            std::cout << "\n";
            ++colIndex;
        }
    }
    // #endif
};
