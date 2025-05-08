#pragma once
#include <cuda_runtime.h>
#include "utils/macros.hpp"

struct NullBitSet {
    char* bitset = nullptr;
    char* bitset_cpu = nullptr;
    // I tried to put as bitset where int*bitset and every number hold 32rows
    // But then I found I need to atomic read & write foreach time as I write in same 32 bit

    NullBitSet(int sz, cudaStream_t stream) {
        CUDA_CHECK( cudaMallocAsync((void**)&bitset, sz, stream));
    }

    NullBitSet(char* h_nullset, int sz, cudaStream_t stream = cudaStreamDefault){
            CUDA_CHECK( cudaMallocAsync((void**)&bitset, sz, stream));
            CUDA_CHECK( cudaMemcpyAsync(bitset, h_nullset, sz, cudaMemcpyHostToDevice, stream) );
    }

    void moveToCpu(int sz, cudaStream_t stream) {
        if(bitset_cpu) return;
        CUDA_CHECK( cudaMallocHost((void**)&bitset_cpu, sz));
        CUDA_CHECK( cudaMemcpyAsync(bitset_cpu, bitset, sz, cudaMemcpyDeviceToHost, stream) );
    }
    // since will delete only when all not needed
    // as it shared_ptr in batch
    ~NullBitSet() {
        if(bitset)
            CUDA_CHECK(cudaFree(bitset));
        if(bitset_cpu)
            CUDA_CHECK(cudaFreeHost(bitset_cpu));
    }
};