#pragma once
#include <nvtx3/nvToolsExt.h>
#include "logger.hpp"
#include <cuda_runtime.h>

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

#define PROFILING_GPU(name, warmup_runs, num_runs, code) { \
    std::vector<long long> durations;                             \
    cudaEvent_t start, end;                                       \
    cudaEventCreate(&start);                                      \
    cudaEventCreate(&end);                                        \
    for (int i = 0; i < warmup_runs; ++i) {                       \
        code;                                                     \
    }                                                             \
    for (int i = 0; i < num_runs; ++i) {                          \
        cudaEventRecord(start, 0);                                \
        code;                                                     \
        cudaEventRecord(end, 0);                                  \
        cudaEventSynchronize(end);                                \
        float duration_ms;                                        \
        cudaEventElapsedTime(&duration_ms, start, end);           \
        durations.push_back(duration_ms);                         \
    }                                                             \
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(100))); \
    cudaEventDestroy(start);                                      \
    cudaEventDestroy(end);                                        \
    std::cout << "\033[1;35m<⏱️  " << name << "> \033[0m" << std::endl; \
    std::sort(durations.begin(), durations.end());                \
    double total = 0;                                             \
    for (auto d : durations) total += d;                          \
    float median = durations[durations.size() / 2];               \
    float min = durations.front();                                \
    float max = durations.back();                                 \
    double mean = total / durations.size();                       \
    double variance = 0;                                          \
    for (auto d : durations) variance += (d - mean) * (d - mean);  \
    variance /= durations.size();                                  \
    std::cout << "\033[0;36mMean: " << mean << "ms, Median: " << median << " ms, Min: " << min << " ms, Max: " << max << " ms, StdDev: " << sqrt(variance) << " ms\033[0m" << std::endl;\
}

// End of file