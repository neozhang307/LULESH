#include "tiled_kernels.cuh"

// DO NOT include tiled_domain.h here!

/**
 * Kernel to copy data across an X-face boundary
 */
__global__ void CopyXFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < ny && z < nz) {
        int srcIdx = srcOffset + y * nx + z * nx * ny;
        int dstIdx = dstOffset + y * nx + z * nx * ny;
        
        dst[dstIdx] = src[srcIdx];
    }
}

/**
 * Kernel to copy data across a Y-face boundary
 */
__global__ void CopyYFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < nx && z < nz) {
        int srcIdx = x + srcOffset * nx + z * nx * ny;
        int dstIdx = x + dstOffset * nx + z * nx * ny;
        
        dst[dstIdx] = src[srcIdx];
    }
}

/**
 * Kernel to copy data across a Z-face boundary
 */
__global__ void CopyZFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < nx && y < ny) {
        int srcIdx = x + y * nx + srcOffset * nx * ny;
        int dstIdx = x + y * nx + dstOffset * nx * ny;
        
        dst[dstIdx] = src[srcIdx];
    }
}

/**
 * Kernel to copy data across an edge
 */
__global__ void CopyEdgeKernel(Real_t* src, Real_t* dst,
                              Int_t srcOffset, Int_t dstOffset,
                              Int_t edgeLength, Int_t edgeType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < edgeLength) {
        dst[dstOffset + i] = src[srcOffset + i];
    }
}

/**
 * Kernel to copy data at a corner
 */
__global__ void CopyCornerKernel(Real_t* src, Real_t* dst,
                                Int_t srcOffset, Int_t dstOffset)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[dstOffset] = src[srcOffset];
    }
}