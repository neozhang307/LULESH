#pragma once

#include <cuda_runtime.h>

// Kernel for initializing an array with a specific value
template<typename T>
__global__ void initArrayKernel(T* array, T value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = value;
    }
}

// MimicFill - A replacement for thrust::fill with stream support
template<typename T>
void MimicFill(T* array, size_t size, T value, cudaStream_t stream = 0) {
    if (value == 0) {
        // For zero initialization, use cudaMemset which is more efficient
        cudaMemsetAsync(array, 0, size * sizeof(T), stream);
    } else {
        // For non-zero values, use a kernel
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        initArrayKernel<<<numBlocks, blockSize, 0, stream>>>(array, value, size);
    }
}