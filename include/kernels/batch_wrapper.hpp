#pragma once
#include <cuda_runtime.h>


struct BatchWrapper {
    void **data;      //ncol * batchSize // array of columns to each pointer data
    unsigned int *bytes;    // array of size of each column
    unsigned int batchSize; // i.e nrows
    unsigned int ncol;
    BatchWrapper() {}
    BatchWrapper(unsigned int ncol, unsigned int batchSize, unsigned int *bytes) : ncol(ncol), batchSize(batchSize), bytes(bytes) {
        //TODO: think later
    }

    // first access of [col]
    inline __device__ void copy_cell(const BatchWrapper* src, unsigned int src_row, unsigned int dist_row, unsigned int col) {
        if(src_row < src->batchSize && dist_row < this->batchSize) {
            char* ptr = static_cast<char*>(data[col]) + dist_row * bytes[col];
            char* src_ptr = static_cast<char*> (src->data[col]) + src_row * bytes[col];
            for(unsigned int i = 0;i < bytes[col];i++) {
                ptr[i] = src_ptr[i];
            }   
        }
    }

    // helper:  get data for certain row, col
    __device__ void* get_data(unsigned int row, unsigned int col) const {
        if (row >= batchSize || col >= ncol) {
            return nullptr; // No exceptions on device, return nullptr
        }
        return static_cast<char*>(data[col]) + row * bytes[col];
    }
    
    //helper: compare rows between two batches, returns -1 (less), 0 (equal), 1 (greater)
    __device__ int compare_row(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        for(unsigned int col = 0; col < ncol; col++) {
            char* l_ptr = static_cast<char*>(data[col]) + row_l * bytes[col];
            char* r_ptr = static_cast<char*>(r->data[col]) + row_r * r->bytes[col];
            
            for(unsigned int i = 0;i < bytes[col];i++) {
                if (l_ptr[i] < r_ptr[i]) return -1; 
                if (l_ptr[i] > r_ptr[i]) return +1;
            }
        }
        return 0; // equal
    } 

    // === comparison operators ===
    __device__ bool less_than(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) < 0;
    }

    __device__ bool greater_than(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) > 0;
    }

    __device__ bool less_equal(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) <= 0;
    }

    __device__ bool greater_equal(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) >= 0;
    }

    __device__ bool equal(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) == 0;
    }

    __device__ bool not_equal(const BatchWrapper* r, unsigned int row_l, unsigned int row_r) const {
        return compare_row(r, row_l, row_r) != 0;
    }
};


