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

#define PROFILING(name, warmup_runs, num_runs, code) {                                              \
    std::vector<long long> durations;                                                               \
    /* Warmup runs */                                                                               \
    for (int i = 0; i < warmup_runs; ++i) {                                                         \
        code;                                                                                       \
    }                                                                                               \
    /* Measured runs */                                                                             \
    for (int i = 0; i < num_runs; ++i) {                                                            \
        auto start = std::chrono::high_resolution_clock::now();                                     \
        code;                                                                                       \
        auto end = std::chrono::high_resolution_clock::now();                                       \
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
        durations.push_back(duration);                                                              \
    }                                                                                               \
    /* Output results */                                                                            \
    std::cout << "\033[1;35m<⏱️  " << name << "> \033[0m" << std::endl;                             \
    for (size_t i = 0; i < durations.size(); ++i) {                                                 \
        std::cout << "\033[0;36mRun " << (i + 1) << ": " << durations[i] << " ms\033[0m" << std::endl; \
    }                                                                                                   \
    if (!durations.empty()) { \
        double total = 0; \
        for (auto d : durations) total += d; \
        std::cout << "\033[0;36mAverage: " << (total / (double)durations.size()) << " ms\033[0m" << std::endl; \
    } \
}
// End of file