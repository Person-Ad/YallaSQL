#pragma once
// forward declaration of data_type.hpp
enum class DataType : __uint8_t;
[[nodiscard]] inline unsigned int getDataTypeNumBytes(DataType type);

namespace YallaSQL
{
    // Maximum string length in bytes
    const unsigned int MAX_STR_LEN = 250;
    // Target maximum bytes per batch (16 KB)
    const unsigned int MAX_BYTES_PER_BATCH = 16 * (1 << 10);

    // const unsigned int DEFAULE_BATCH_SIZE = 32;
    
    // Default alignment for GPU operations (32 is common for many CUDA operations)
    const unsigned int GPU_ALIGNMENT = 32;
    inline unsigned int calculateOptimalBatchSize(const std::vector<DataType>& columnTypes) {
        // calculate row size
        unsigned int bytesPerRow = 0;
        for (const auto& type : columnTypes)  bytesPerRow += getDataTypeNumBytes(type);
        // Fallback to minimum batch size
        if (bytesPerRow <= GPU_ALIGNMENT) 
            return GPU_ALIGNMENT; 

        // Calculate maximum batch size that fits within MAX_BYTES_PER_BATCH
        unsigned int maxBatchSize = MAX_BYTES_PER_BATCH / bytesPerRow;
        // make it multiple of GPU_ALIGNMENT
        unsigned int alignedBatchSize = (maxBatchSize / GPU_ALIGNMENT) * GPU_ALIGNMENT;

        return std::max(GPU_ALIGNMENT, alignedBatchSize);
    }
} // namespace YallaSQL
