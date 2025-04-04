#ifndef LULESH_COMM_H
#define LULESH_COMM_H

#include "lulesh_split.h"

// Function prototypes for communication components
void LagrangeNodal(Domain *domain);
void LagrangeElements(Domain *domain);
void CalcForceForNodes(Domain *domain);
void TimeIncrement(Domain* domain);

// These are implemented in lulesh_compute.cu but needed by lulesh_compute_comm.cu
void CalcAccelerationForNodes(Domain *domain);
void CalcPositionAndVelocityForNodes(const Real_t u_cut, Domain* domain);
void CalcKinematicsAndMonotonicQGradient(Domain *domain);
void ApplyMaterialPropertiesAndUpdateVolume(Domain *domain);
void ApplyAccelerationBoundaryConditionsForNodes(Domain *domain);
void CalcVolumeForceForElems(Domain* domain);
void CalcMonotonicQRegionForElems(Domain *domain);

// Kernel declarations needed by both files
extern __global__ void CalcAccelerationForNodes_kernel(int numNode,
    Real_t* xdd, Real_t* ydd, Real_t* zdd,
    const Real_t* fx, const Real_t* fy, const Real_t* fz,
    const Real_t* nodalMass);

extern __global__ void ApplyAccelerationBoundaryConditionsForNodes_kernel(int numNodes,
    Real_t* dd, Index_t* nodes);

extern __global__ void CalcPositionAndVelocityForNodes_kernel(int numNode,
    const Real_t deltatime,
    const Real_t u_cut,
    Real_t* x,  Real_t* y,  Real_t* z,
    Real_t* xd, Real_t* yd, Real_t* zd,
    const Real_t* xdd, const Real_t* ydd, const Real_t* zdd);

#endif // LULESH_COMM_H