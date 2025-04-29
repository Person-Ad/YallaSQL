#pragma once
#include <string>
#include <vector> // For std::vector
#include <cstddef> // For std::size_t
// forward declaration of data_type.hpp
enum class DataType : __uint8_t;
[[nodiscard]] inline unsigned int getDataTypeNumBytes(DataType type);

namespace YallaSQL
{
    // Maximum string length in bytes
    constexpr size_t MAX_STR_LEN = 250;
    // Target maximum bytes per batch (32 mb)
    constexpr size_t  MAX_BYTES_PER_BATCH = 32 * (1ULL << 20);
    // Target maximum bytes per batch (6 gb)
    constexpr size_t  MAX_LIMIT_CPU_CACHE = 6 * (1ULL << 30);
    // Target maximum bytes per batch (3 gb)
    constexpr size_t  MAX_LIMIT_GPU_CACHE = 3 * (1ULL << 30);


    const std::string cacheDir = ".cache";
    const std::string resultDir = "results";

    // const unsigned int DEFAULE_BATCH_SIZE = 32;
    
    // Default alignment for GPU operations (32 is common for many CUDA operations)
    constexpr unsigned int GPU_ALIGNMENT = 32;
    inline unsigned int calculateOptimalBatchSize(const std::vector<DataType>& columnTypes) {
        // calculate row size
        size_t bytesPerRow = 0;
        for (const auto& type : columnTypes)  bytesPerRow += getDataTypeNumBytes(type);
        // Fallback to minimum batch size
        if (bytesPerRow <= GPU_ALIGNMENT) 
            return GPU_ALIGNMENT; 

        // Calculate maximum batch size that fits within MAX_BYTES_PER_BATCH
        size_t maxBatchSize = MAX_BYTES_PER_BATCH / bytesPerRow;
        // make it multiple of GPU_ALIGNMENT
        unsigned int alignedBatchSize = (maxBatchSize / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        return std::max(GPU_ALIGNMENT, alignedBatchSize);
    }
} // namespace YallaSQL
