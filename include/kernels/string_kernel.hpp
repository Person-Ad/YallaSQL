#pragma once
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {
    struct __align__(8) String {
        char data[MAX_STR];

        // Ensure null-termination
        __host__ __device__ String() {
            #pragma unroll
            for(int i = 0;i < MAX_STR;i++)
                data[i] = '\0';
        }

        __host__ __device__ void set(const char *src) {
            if (!src) {
                data[0] = '\0';
                return;
            }

            size_t i = 0;
            while (i < MAX_STR - 1 && src[i] != '\0') {
                data[i] = src[i];
                i++;
            }
            // for sorting and other operations
            for (; i < MAX_STR; i++)
                data[i] = '\0';
        }
        //REFERENCE FROM HERE GROK
        // Right-shift operator (>>) // used in radix sort with & to get bit 
        __host__ __device__ unsigned int operator>>(unsigned int shift) const {
            unsigned int byte_index = shift / 8;
            unsigned int bit_offset = shift % 8;
            if (byte_index >= MAX_STR) {
                return 0;
            }
            unsigned char byte = (unsigned char)data[byte_index];
            return (byte >> bit_offset) & 0xFF; // Mask to 8 bits
        }

        
    };

    inline __device__ int strcmp_device(const String& s1, const String& s2) {
        uint32_t i = 0;
        while (i < MAX_STR && s1.data[i] && (s1.data[i] == s2.data[i])) {
            i++;
        }
        // If we reached end of sz
        if (i >= MAX_STR) 
            return 0;
        
        return (unsigned char)s1.data[i] - (unsigned char)s2.data[i];
    }
}