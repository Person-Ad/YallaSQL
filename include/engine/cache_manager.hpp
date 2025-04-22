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
        cache.erase(batchId);
        return std::move(batch);
    }
    // put batch of data in the cache
    BatchID putBatch(std::unique_ptr<Batch> batch) {
        // TODO: if we added pipeline we need to add mutex lock
        if (!batch)
            throw std::invalid_argument("Invalid batch provided.");

        cache[incrementId] = std::move(batch);

        if(cache[incrementId]->location == Device::CPU) {
            cpuBytes += cache[incrementId]->totalBytes;
            //TODO: handle if it exceed MAX_LIMIT_CPU_CACHE by dumping in file
        } else {
            gpuBytes += cache[incrementId]->totalBytes;
            //TODO: handle if it exceed MAX_LIMIT_GPU_CACHE by dumping to cpu if exceed limit
        }

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
};
