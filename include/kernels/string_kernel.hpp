#pragma once
#include "kernels/constants.hpp"
#include <cuda_runtime.h>

namespace YallaSQL::Kernel {
    struct String {
        static constexpr size_t MAX_SIZE = 250;
        char data[MAX_SIZE];

        // Ensure null-termination
        __host__ __device__ String() {
            data[0] = '\0';
        }

        __host__ __device__ void set(const char *src) {
            if(!src) {
                data[0] = '\0';
                return;
            }

            size_t i = 0;
            while (i < MAX_SIZE - 1 && src[i] != '\0') {
                data[i] = src[i];
                i++;
            }
            data[i] = '\0';
        }
    };


    inline __device__ int strcmp_device(const String& s1, const String& s2) {
        uint32_t i = 0;
        while (i < String::MAX_SIZE && s1.data[i] && (s1.data[i] == s2.data[i])) {
            i++;
        }
        // if we reached end of sz
        if (i >= String::MAX_SIZE) 
            return 0;
        
    
        return (unsigned char)s1.data[i] - (unsigned char)s2.data[i];
    }
}