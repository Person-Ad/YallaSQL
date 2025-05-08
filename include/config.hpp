#pragma once
#include <string>
#include <vector> // For std::vector
#include <cstddef> // For std::size_t
#include "kernels/constants.hpp"
// forward declaration of data_type.hpp
enum class DataType : __uint8_t;
[[nodiscard]] inline unsigned int getDataTypeNumBytes(DataType type);

namespace YallaSQL
{
    // Maximum string length in bytes
    constexpr size_t MAX_STR_LEN = MAX_STR;
    // Target maximum bytes per batch (16 mb)
    constexpr size_t  MAX_BYTES_PER_BATCH = 2 * (1ULL << 20);
    // Target maximum bytes per batch (6 gb)
    constexpr size_t  MAX_LIMIT_CPU_CACHE = 2 * (1ULL << 30);
    // Target maximum bytes per batch (3 gb)
    constexpr size_t  MAX_LIMIT_GPU_CACHE = 1 * (1ULL << 30);

    constexpr size_t MAX_ROWS_OUT_JOIN_OP = 1000000;

    const std::string cacheDir = ".cache";
    const std::string resultDir = "results";

    // const unsigned int DEFAULE_BATCH_SIZE = 32;
    
    // Default alignment for GPU operations (32 is common for many CUDA operations)
    constexpr unsigned int GPU_ALIGNMENT = BLOCK_DIM;
    
    inline unsigned int calculateOptimalBatchSize(const std::vector<DataType>& columnTypes) {
        // calculate row size
        size_t bytesPerRow = 0;
        for (const auto& type : columnTypes)  bytesPerRow += getDataTypeNumBytes(type);
        // Calculate maximum batch size that fits within MAX_BYTES_PER_BATCH
        size_t maxBatchSize = MAX_BYTES_PER_BATCH / bytesPerRow;
        // make it multiple of GPU_ALIGNMENT
        unsigned int alignedBatchSize = (maxBatchSize / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        return std::max(GPU_ALIGNMENT, alignedBatchSize);
    }
} // namespace YallaSQL
