#pragma once
#include <stddef.h>
#include <unordered_map>
#include "batch.hpp"

#define u_int32 unsigned int
#define NodeID unsigned int
#define BatchID unsigned int

class CacheManager {
public:
    // get batch of data
    std::unique_ptr<Batch> getBatch(BatchID batchId) {
        std::unique_ptr<Batch> batch = std::move(cache[batchId]);
        if (batch->location == Device::GPU) {
            gpuBytes -= batch->totalBytes;
        } else if (batch->location == Device::CPU) {
            cpuBytes -= batch->totalBytes;
        }
        cache.erase(batchId);
        return std::move(batch);
    }
    Batch& refBatch (BatchID batchId) {
        return *cache[batchId];
    }

    // put batch of data in the cache
    BatchID putBatch(std::unique_ptr<Batch> batch) {
        // TODO: if we added pipeline we need to add mutex lock
        if (!batch)
            throw std::invalid_argument("Invalid batch provided.");

        cache[incrementId] = std::move(batch);

        if(cache[incrementId]->location == Device::CPU)
            cpuBytes += cache[incrementId]->totalBytes;
        else
            gpuBytes += cache[incrementId]->totalBytes;


        if (cpuBytes > YallaSQL::MAX_LIMIT_CPU_CACHE) handleOverflow(Device::CPU);
        if (gpuBytes > YallaSQL::MAX_LIMIT_GPU_CACHE) handleOverflow(Device::GPU);

        return incrementId++;
    }

    // clean data after finishing 
    // since it's unique_ptr there's no need to delete it
    ~CacheManager() {
    }
private:
    // incremental batch id - start from 1 as 0 used as empty batch
    BatchID incrementId = 1; 
    // size of bytes on gpu
    u_int32 gpuBytes = 0;
    // size of bytes on cpu
    u_int32 cpuBytes = 0;
    
    std::unordered_map<BatchID, std::unique_ptr<Batch>> cache;
    // handle overflows
    // note since we use unique_ptr & everyone take ownership of batch
    // we can use BatchId as our last access
    void handleOverflow(Device device) {
        // auto& [id, batch] = *(cache.begin());
        BatchID id = cache.begin()->first;
        Batch*   batch = cache.begin()->second.get();
        for (auto& [batchId, batchPtr] : cache) {
            if (batchPtr->location == device && batchId < id) {
                id = batchId;
                batch = batchPtr.get();
            }
        }

        // Sort by last accessed time (oldest first)
        if (device == Device::GPU) {
            batch->moveTo(Device::CPU);
            gpuBytes -= batch->totalBytes;
            cpuBytes += batch->totalBytes;
        } else if (device == Device::CPU) {
            batch->moveTo(Device::FS);
            cpuBytes -= batch->totalBytes;
        }
    }
};
