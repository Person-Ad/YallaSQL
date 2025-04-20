#pragma once
#include <stddef.h>
#include <unordered_map>
#include "batch.cu"

#define u_int32 unsigned int
#define NodeID unsigned int
#define BatchID unsigned int

class CacheManager {
public:
    // get batch of data
    Batch* getBatch(BatchID batchId, Device preferredDevice) {
        Batch* batch = cache[batchId];
        // move it to better place
        if(batch->location != preferredDevice) 
            batch->moveTo( preferredDevice );
        //
        return batch;
    }
    // put batch of data in cache
    BatchID putBatch(Batch* batch) {
        // TODO: if we added pipeline we need to add mutex lock
        cache[incrementId] = batch;

        if(batch->location == Device::CPU) {
            cpuBytes += batch->bytesSum.back() * batch->batchSize;
        } else {
            gpuBytes += batch->bytesSum.back() * batch->batchSize;
            //TODO: handle if it exceed MAX_LIMIT_GPU_CACHE
        }

        return incrementId++;
    }
    // clean data after finishing
    ~CacheManager() {
        for(auto [_, batch]: cache) {
            if(batch) delete batch;
        }
    }
private:
    // incremental batch id - start from 1 as 0 used as empty batch
    BatchID incrementId = 1; 
    // size of bytes on gpu
    u_int32 gpuBytes = 0;
    // size of bytes on cpu
    u_int32 cpuBytes = 0;
    
    
    std::unordered_map<BatchID, Batch*> cache;
};
