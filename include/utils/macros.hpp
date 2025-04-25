#pragma once

#include "logger.hpp"
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR(YallaSQL::getLogger(""), "CUDA error in {} at line {}: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUDA_CHECK_LAST() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        LOG_ERROR(YallaSQL::getLogger(""), "CUDA error in {} at line {}: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CEIL_DIV(X, Y) (X + Y - 1)/(Y)