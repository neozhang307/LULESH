#ifndef __TILED_KERNELS_CUH__
#define __TILED_KERNELS_CUH__

#include "../split/lulesh_split.h"

// Kernel for copying an X-face boundary
extern __global__ void CopyXFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize);

// Kernel for copying a Y-face boundary
extern __global__ void CopyYFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize);

// Kernel for copying a Z-face boundary
extern __global__ void CopyZFaceKernel(Real_t* src, Real_t* dst,
                               Int_t srcOffset, Int_t dstOffset,
                               Int_t nx, Int_t ny, Int_t nz,
                               Int_t planeSize);

// Kernel for copying an edge
extern __global__ void CopyEdgeKernel(Real_t* src, Real_t* dst,
                              Int_t srcOffset, Int_t dstOffset,
                              Int_t edgeLength, Int_t edgeType);

// Kernel for copying a corner
extern __global__ void CopyCornerKernel(Real_t* src, Real_t* dst,
                                Int_t srcOffset, Int_t dstOffset);

#endif // __TILED_KERNELS_CUH__