#pragma once
#include <deque>
#include <future>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <charconv>
#include "batch.hpp"
#include "enums/data_type.hpp"

class CsvWriter {
private:
    static constexpr size_t BUFFER_SIZE = YallaSQL::MAX_BYTES_PER_BATCH * 5; // 5x the max batch size
    std::string filePath;
    std::ofstream ofs;
    std::deque<std::string> batchQueue;
    std::mutex queueMutex;
    std::future<void> writerFuture;
    bool running;
    bool headerWritten;
    std::string writeBuffer;
    size_t writeBufferUsed;

    // Append to write buffer, flushing to disk if full
    void appendToBuffer(std::string_view data) {
        if (writeBufferUsed + data.size() > BUFFER_SIZE) {
            flushBuffer();
        }
        writeBuffer.append(data.data(), data.size());
        writeBufferUsed += data.size();
    }

    // Flush write buffer to disk
    void flushBuffer() {
        if (writeBufferUsed > 0) {
            ofs.write(writeBuffer.data(), writeBufferUsed);
            writeBuffer.clear();
            writeBufferUsed = 0;
        }
    }

    // Convert a batch to CSV and append to buffer
    void batchToCsv(const Batch& batch) {
        // Preallocate buffer for this batch (rough estimate)
        std::string csvData;
        // approaximate preallocation of buffer
        csvData.reserve(batch.totalBytes);

        // Write header if not yet written
        if (!headerWritten) {
            for (size_t i = 0; i < batch.columns.size(); ++i) {
                std::string_view name = batch.columns[i]->name;
                csvData.append(name.data(), name.size());
                if (i < batch.columns.size() - 1) csvData += ',';
            }
            csvData += '\n';
            headerWritten = true;
        }

        // Write rows
        char numBuffer[32]; // Buffer for numeric conversions
        for (uint32_t row = 0; row < batch.batchSize; ++row) {
            for (size_t col = 0; col < batch.columns.size(); ++col) {
                auto column = batch.columns[col];

                if (column->type == DataType::STRING) {
                    auto* value = batch.getItem<const char>(col, row);
                    csvData += '"';
                    csvData += value; // Assumes null-terminated
                    csvData += '"';
                } else if (column->type == DataType::INT) {
                    int value = *batch.getItem<int>(col, row);
                    auto [ptr, ec] = std::to_chars(numBuffer, numBuffer + sizeof(numBuffer), value);
                    csvData.append(numBuffer, ptr - numBuffer);
                } else if (column->type == DataType::FLOAT) {
                    float value = *batch.getItem<float>(col, row);
                    auto [ptr, ec] = std::to_chars(numBuffer, numBuffer + sizeof(numBuffer), value, std::chars_format::fixed, 4);
                    csvData.append(numBuffer, ptr - numBuffer);
                } else if (column->type == DataType::DATETIME) {
                    int64_t value = *batch.getItem<int64_t>(col, row);
                    std::string dateStr = getDateTimeStr(value); // Assumes this returns a string
                    csvData += '"';
                    csvData.append(dateStr.data(), dateStr.size());
                    csvData += '"';
                }

                if (col < batch.columns.size() - 1) csvData += ',';
            }
            csvData += '\n';
        }

        appendToBuffer(csvData);
    }

    // Async writer loop
    void writeLoop() {
        writeBuffer.reserve(BUFFER_SIZE);
        while (running) {
            std::vector<std::string> batches;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                if (batchQueue.empty()) {
                    if (!running) break;
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                // Process multiple batches to reduce lock contention
                while (!batchQueue.empty() && batches.size() < 10) {
                    batches.push_back(std::move(batchQueue.front()));
                    batchQueue.pop_front();
                }
            }

            for (const auto& csvData : batches) {
                appendToBuffer(csvData);
            }
        }

        // Write any remaining data
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (batchQueue.empty()) break;
            std::string csvData = std::move(batchQueue.front());
            batchQueue.pop_front();
            lock.unlock();
            appendToBuffer(csvData);
        }

        flushBuffer();
    }

public:
    CsvWriter(const std::string& path) : filePath(path), running(true), headerWritten(false), writeBufferUsed(0) {
        ofs.open(filePath, std::ios::out | std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + filePath);
        }
        writerFuture = std::async(std::launch::async, &CsvWriter::writeLoop, this);
    }

    ~CsvWriter() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            running = false;
        }
        writerFuture.wait();
        flushBuffer();
        ofs.close();
    }

    // Add a batch to the queue for writing
    void addBatch(const Batch& batch) {
        if (batch.location != Device::CPU) {
            throw std::runtime_error("Batch must be on CPU for CSV writing");
        }

        std::string csvData;
        {
            // Lock to ensure headerWritten is consistent
            std::unique_lock<std::mutex> lock(queueMutex);
            batchToCsv(batch);
            batchQueue.push_back(std::move(writeBuffer));
            writeBuffer.clear();
            writeBufferUsed = 0;
        }
    }
};