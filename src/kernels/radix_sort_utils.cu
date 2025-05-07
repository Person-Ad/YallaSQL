#include "kernels/radix_sort_kernel.hpp"
namespace YallaSQL::Kernel {

__global__  void int_to_uint32(int* arr, uint32_t* res, int N) {
    // Flip the sign bit to make negative numbers come before positive ones in unsigned space
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        res[global_idx] = static_cast<uint32_t>(arr[global_idx]) ^ 0x80000000;
    }
}

__global__  void uint32_to_int(uint32_t* arr, int* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        res[global_idx] = static_cast<int32_t>(arr[global_idx] ^ 0x80000000);
    }
}

__global__  void float_to_uint32(float* arr, uint32_t* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&arr[global_idx]);
        res[global_idx] = (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
    }
}

__global__  void uint32_to_float(uint32_t* arr, float* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        uint32_t x = arr[global_idx];
        uint32_t bits = (x & 0x80000000) ? (x ^ 0x80000000) : ~x;
        res[global_idx] = *reinterpret_cast<float*>(&bits);
    }
}

void launch_int_to_uint32(int* arr, uint32_t* res, int N) {
    int_to_uint32<<<CEIL_DIVI(N, BLOCK_DIM), BLOCK_DIM>>>(arr, res, N);
}

void launch_uint32_to_int(uint32_t* arr, int* res, int N) {
    uint32_to_int<<<CEIL_DIVI(N, BLOCK_DIM), BLOCK_DIM>>>(arr, res, N);
}

void launch_float_to_uint32(float* arr, uint32_t* res, int N) {
    float_to_uint32<<<CEIL_DIVI(N, BLOCK_DIM), BLOCK_DIM>>>(arr, res, N);
}

void launch_uint32_to_float(uint32_t* arr, float* res, int N) {
    uint32_to_float<<<CEIL_DIVI(N, BLOCK_DIM), BLOCK_DIM>>>(arr, res, N);
}

}