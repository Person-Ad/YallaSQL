#pragma once
#include <deque>
#include <future>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <charconv>
#include "batch.hpp"
#include "enums/data_type_utils.hpp"

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

    // Convert a batch to CSV string
    std::string batchToCsv(const Batch& batch) {
        std::string csvData;
        csvData.reserve(batch.totalBytes); // Approximate preallocation

        // Write header if not yet written
        {
            std::lock_guard<std::mutex> lock(queueMutex); // Protect headerWritten
            if (!headerWritten) {
                for (size_t i = 0; i < batch.columns.size(); ++i) {
                    std::string_view name = batch.columns[i]->name;
                    csvData.append(name.data(), name.size());
                    if (i < batch.columns.size() - 1) csvData += ',';
                }
                csvData += '\n';
                headerWritten = true;
            }
        }

        // Write rows
        char numBuffer[32]; // Buffer for numeric conversions
        for (uint32_t row = 0; row < batch.batchSize; ++row) {
            for (size_t col = 0; col < batch.columns.size(); ++col) {
                auto column = batch.columns[col];
                if(batch.nullset[col]->bitset_cpu[row]) { //* MUST Move To CPU Before
                    csvData += "\"\"";
                }
                else if (column->type == DataType::STRING) {
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

        return csvData;
    }

    // Async writer loop
    void writeLoop() {
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

            // Write batches to file
            for (const auto& csvData : batches) {
                ofs.write(csvData.data(), csvData.size());
            }
            ofs.flush(); // Ensure data is written to disk
        }

        // Write any remaining data
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (batchQueue.empty()) break;
            std::string csvData = std::move(batchQueue.front());
            batchQueue.pop_front();
            lock.unlock();
            ofs.write(csvData.data(), csvData.size());
        }

        ofs.flush(); // Final flush
    }

public:
    CsvWriter(const std::string& path) : filePath(path), running(true), headerWritten(false) {
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
        ofs.close();
    }

    // Add a batch to the queue for writing
    void addBatch(const Batch& batch) {
        if (batch.location != Device::CPU) {
            throw std::runtime_error("Batch must be on CPU for CSV writing");
        }

        std::string csvData = batchToCsv(batch);
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            batchQueue.push_back(std::move(csvData));
        }
    }
};