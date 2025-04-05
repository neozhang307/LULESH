/*
                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "lulesh_split.h"
#include "utility/util.h"
#include "utility/sm_utils.inl"
// #include "utility/allocator.h"
#include "lulesh_kernels.h"
#include "tiling_utils.h"
// checkErrors function is already defined at line 75






void CalcVolumeForceForElems(const Real_t hgcoef, Domain *domain, cudaStream_t stream)
{
    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

#ifdef DOUBLE_PRECISION
    Real_t* fx_elem;
    Real_t* fy_elem;
    Real_t* fz_elem;
    cudaMallocAsync((void**)&fx_elem, padded_numElem*8*sizeof(Real_t), stream);
    cudaMallocAsync((void**)&fy_elem, padded_numElem*8*sizeof(Real_t), stream);
    cudaMallocAsync((void**)&fz_elem, padded_numElem*8*sizeof(Real_t), stream);
#else
    MimicFill(domain->fx, domain->numNode, Real_t(0.), stream);
    MimicFill(domain->fy, domain->numNode, Real_t(0.), stream);
    MimicFill(domain->fz, domain->numNode, Real_t(0.), stream);
#endif

    int num_threads = numElem ;
    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);

    bool hourg_gt_zero = hgcoef > Real_t(0.0);
    if (hourg_gt_zero)
    {
      CalcVolumeForceForElems_kernel<true> <<<dimGrid,block_size,0,stream>>>
      ( domain->volo, 
        domain->v, 
        domain->p, 
        domain->q,
	      hgcoef, numElem, padded_numElem,
        domain->nodelist, 
        domain->ss, 
        domain->elemMass,
        domain->x, domain->y, domain->z, domain->xd, domain->yd, domain->zd,
#ifdef DOUBLE_PRECISION
        fx_elem, 
        fy_elem, 
        fz_elem ,
#else
        domain->fx,
        domain->fy,
        domain->fz,
#endif
        domain->bad_vol_h,
        num_threads
      );
    }
    else
    {
      CalcVolumeForceForElems_kernel<false> <<<dimGrid,block_size,0,stream>>>
      ( domain->volo,
        domain->v, 
        domain->p, 
        domain->q,
	      hgcoef, numElem, padded_numElem,
        domain->nodelist, 
        domain->ss, 
        domain->elemMass,
        domain->x, domain->y, domain->z, domain->xd, domain->yd, domain->zd,
#ifdef DOUBLE_PRECISION
        fx_elem, 
        fy_elem, 
        fz_elem ,
#else
        domain->fx,
        domain->fy,
        domain->fz,
#endif
        domain->bad_vol_h,
        num_threads
      );
    }

#ifdef DOUBLE_PRECISION
    num_threads = domain->numNode;

    // Launch boundary nodes first
    dimGrid= PAD_DIV(num_threads,block_size);

    AddNodeForcesFromElems_kernel<<<dimGrid,block_size,0,stream>>>
    ( domain->numNode,
      domain->padded_numNode,
      domain->nodeElemCount,
      domain->nodeElemStart,
      domain->nodeElemCornerList,
#ifdef DOUBLE_PRECISION
      fx_elem,
      fy_elem,
      fz_elem,
#else
      domain->fx,
      domain->fy,
      domain->fz,
#endif
      domain->fx,
      domain->fy,
      domain->fz,
      num_threads
    );

    cudaFreeAsync(fx_elem, stream);
    cudaFreeAsync(fy_elem, stream);
    cudaFreeAsync(fz_elem, stream);

#endif // ifdef DOUBLE_PRECISION
   return ;
}


// Remove static qualifier for function used across files
void CalcVolumeForceForElems(Domain* domain, cudaStream_t stream)
{
      const Real_t hgcoef = domain->hgcoef ;

     CalcVolumeForceForElems(hgcoef,domain, stream);

     //CalcVolumeForceForElems_warp_per_4cell(hgcoef,domain);
}




// static inline
void CalcAccelerationForNodes(Domain *domain, cudaStream_t stream)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcAccelerationForNodes_kernel<<<dimGrid, dimBlock,0,stream>>>
        (domain->numNode,
         domain->xdd,domain->ydd,domain->zdd,
         domain->fx,domain->fy,domain->fz,
         domain->nodalMass);

    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void ApplyAccelerationBoundaryConditionsForNodes(Domain *domain, cudaStream_t stream)
{

    Index_t dimBlock = 128;

    Index_t dimGrid = PAD_DIV(domain->numSymmX,dimBlock);
    if (domain->numSymmX > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock,0,stream>>>
        (domain->numSymmX,
         domain->xdd,
         domain->symmX);

    dimGrid = PAD_DIV(domain->numSymmY,dimBlock);
    if (domain->numSymmY > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock,0,stream>>>
        (domain->numSymmY,
         domain->ydd,
         domain->symmY);

    dimGrid = PAD_DIV(domain->numSymmZ,dimBlock);
    if (domain->numSymmZ > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock,0,stream>>>
        (domain->numSymmZ,
         domain->zdd,
         domain->symmZ);
}

__global__
void CalcPositionAndVelocityForNodes_kernel(int numNode, 
    const Real_t deltatime, 
    const Real_t u_cut,
    Real_t* __restrict__ x,  Real_t* __restrict__ y,  Real_t* __restrict__ z,
    Real_t* __restrict__ xd, Real_t* __restrict__ yd, Real_t* __restrict__ zd,
    const Real_t* __restrict__ xdd, const Real_t* __restrict__ ydd, const Real_t* __restrict__ zdd)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i < numNode)
    {
      Real_t xdtmp, ydtmp, zdtmp, dt;
      dt = deltatime;
      
      xdtmp = xd[i] + xdd[i] * dt ;
      ydtmp = yd[i] + ydd[i] * dt ;
      zdtmp = zd[i] + zdd[i] * dt ;

      if( FABS(xdtmp) < u_cut ) xdtmp = 0.0;
      if( FABS(ydtmp) < u_cut ) ydtmp = 0.0;
      if( FABS(zdtmp) < u_cut ) zdtmp = 0.0;

      x[i] += xdtmp * dt;
      y[i] += ydtmp * dt;
      z[i] += zdtmp * dt;
      
      xd[i] = xdtmp; 
      yd[i] = ydtmp; 
      zd[i] = zdtmp; 
    }
}

void CalcPositionAndVelocityForNodes(const Real_t u_cut, Domain* domain, cudaStream_t stream)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcPositionAndVelocityForNodes_kernel<<<dimGrid, dimBlock,0,stream>>>
        (domain->numNode,domain->deltatime_h,u_cut,
         domain->x,domain->y,domain->z,
         domain->xd,domain->yd,domain->zd,
         domain->xdd,domain->ydd,domain->zdd);

    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void CalcKinematicsAndMonotonicQGradient(Domain *domain, cudaStream_t stream)
{
    printf("DEBUG: Inside CalcKinematicsAndMonotonicQGradient\n");
    fflush(stdout);

    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

    int num_threads = numElem;

    printf("DEBUG: Preparing kernel for %d elements (%d padded)\n", numElem, padded_numElem);
    fflush(stdout);

    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);
    
    printf("DEBUG: Launching kernel with %d blocks of %d threads each\n", dimGrid, block_size);
    fflush(stdout);
    
    // Debugging - Check key pointers before kernel launch
    if (domain->volo == nullptr) {
        printf("ERROR: domain->volo is NULL inside CalcKinematicsAndMonotonicQGradient\n");
        fflush(stdout);
    }
    
    if (domain->vnew == nullptr) {
        printf("ERROR: domain->vnew is NULL inside CalcKinematicsAndMonotonicQGradient\n");
        fflush(stdout);
    }
    
    // Read the first element of volo for debugging
    Real_t firstVolo;
    cudaError_t err = cudaMemcpyAsync(&firstVolo, domain->volo, sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to read volo[0] inside CalcKinematicsAndMonotonicQGradient: %s\n", 
               cudaGetErrorString(err));
    } else {
        printf("DEBUG: volo[0] = %e before kernel\n", firstVolo);
    }
    fflush(stdout);

    CalcKinematicsAndMonotonicQGradient_kernel<<<dimGrid,block_size,0,stream>>>
    (  numElem,padded_numElem, domain->deltatime_h, 
       domain->nodelist,
       domain->volo,
       domain->v,
       domain->x, domain->y, domain->z, domain->xd, domain->yd, domain->zd,
       domain->vnew,
       domain->delv,
       domain->arealg,
       domain->dxx,
       domain->dyy,
       domain->dzz,
       domain->vdov, 
       domain->delx_zeta,
       domain->delv_zeta, 
       domain->delx_xi, 
       domain->delv_xi,  
       domain->delx_eta, 
       domain->delv_eta,
       domain->bad_vol_h,
       num_threads  
    );

    // Wait for the kernel to complete to check for errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("ERROR: CalcKinematicsAndMonotonicQGradient kernel failed: %s\n", 
               cudaGetErrorString(kernelError));
        fflush(stdout);
    } else {
        printf("DEBUG: CalcKinematicsAndMonotonicQGradient kernel launched successfully\n");
        fflush(stdout);
    }
    
    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void CalcMonotonicQRegionForElems(Domain *domain, cudaStream_t stream)
{
    printf("DEBUG: Inside CalcMonotonicQRegionForElems\n");
    fflush(stdout);

    const Real_t ptiny        = Real_t(1.e-36) ;
    Real_t monoq_max_slope    = domain->monoq_max_slope ;
    Real_t monoq_limiter_mult = domain->monoq_limiter_mult ;

    Real_t qlc_monoq = domain->qlc_monoq;
    Real_t qqc_monoq = domain->qqc_monoq;
    Index_t elength = domain->numElem;

    printf("DEBUG: Parameters - max_slope=%e, limiter_mult=%e, qlc_monoq=%e, qqc_monoq=%e\n", 
           monoq_max_slope, monoq_limiter_mult, qlc_monoq, qqc_monoq);
    fflush(stdout);

    Index_t dimBlock= 128;
    Index_t dimGrid = PAD_DIV(elength,dimBlock);

    printf("DEBUG: Launching kernel with %d blocks of %d threads each\n", dimGrid, dimBlock);
    fflush(stdout);
    
    // Check key pointers before kernel launch
    if (domain->volo == nullptr) {
        printf("ERROR: domain->volo is NULL inside CalcMonotonicQRegionForElems\n");
        fflush(stdout);
    }
    
    // Read the first element of volo for debugging
    Real_t firstVolo;
    cudaError_t err = cudaMemcpyAsync(&firstVolo, domain->volo, sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to read volo[0] inside CalcMonotonicQRegionForElems: %s\n", 
               cudaGetErrorString(err));
    } else {
        printf("DEBUG: volo[0] = %e before MonotonicQ kernel\n", firstVolo);
    }
    fflush(stdout);

    CalcMonotonicQRegionForElems_kernel<<<dimGrid,dimBlock,0,stream>>>
    ( qlc_monoq,qqc_monoq,monoq_limiter_mult,monoq_max_slope,ptiny,elength,
      domain->regElemlist,domain->elemBC,
      domain->lxim,domain->lxip,
      domain->letam,domain->letap,
      domain->lzetam,domain->lzetap,
      domain->delv_xi,domain->delv_eta,domain->delv_zeta,
      domain->delx_xi,domain->delx_eta,domain->delx_zeta,
      domain->vdov,domain->elemMass,domain->volo,domain->vnew,
      domain->qq,domain->ql, 
      domain->q,
      domain->qstop,
      domain->bad_q_h
    );

    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("ERROR: CalcMonotonicQRegionForElems kernel failed: %s\n", 
               cudaGetErrorString(kernelError));
    } else {
        printf("DEBUG: CalcMonotonicQRegionForElems kernel launched successfully\n");
    }
    fflush(stdout);

    //cudaDeviceSynchronize();
    //cudaCheckError();
}






void ApplyMaterialPropertiesAndUpdateVolume(Domain *domain, cudaStream_t stream)
{
  printf("DEBUG: Inside ApplyMaterialPropertiesAndUpdateVolume\n");
  fflush(stdout);
  
  Index_t length = domain->numElem ;

  if (length != 0) {
    printf("DEBUG: Preparing to apply material properties for %d elements\n", length);
    fflush(stdout);

    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(length,dimBlock);
    
    printf("DEBUG: Launching kernel with %d blocks of %d threads each\n", dimGrid, dimBlock);
    fflush(stdout);
    
    // Check key pointers before kernel launch
    if (domain->volo == nullptr) {
        printf("ERROR: domain->volo is NULL inside ApplyMaterialPropertiesAndUpdateVolume\n");
        fflush(stdout);
    }
    
    if (domain->vnew == nullptr) {
        printf("ERROR: domain->vnew is NULL inside ApplyMaterialPropertiesAndUpdateVolume\n");
        fflush(stdout);
    }
    
    // Verify material parameters
    printf("DEBUG: Material parameters - refdens=%e, e_cut=%e, emin=%e, pmin=%e, p_cut=%e, q_cut=%e\n",
           domain->refdens, domain->e_cut, domain->emin, domain->pmin, domain->p_cut, domain->q_cut);
    printf("DEBUG: Volume parameters - eosvmin=%e, eosvmax=%e, v_cut=%e\n",
           domain->eosvmin, domain->eosvmax, domain->v_cut);
    fflush(stdout);

    ApplyMaterialPropertiesAndUpdateVolume_kernel<<<dimGrid,dimBlock,0,stream>>>
        (length,
         domain->refdens,
         domain->e_cut,
         domain->emin,
         domain->ql,
         domain->qq,
         domain->vnew,
         domain->v,
         domain->pmin,
         domain->p_cut,
         domain->q_cut,
         domain->eosvmin,
         domain->eosvmax,
         domain->regElemlist,
         domain->e,
         domain->delv,
         domain->p,
         domain->q,
         domain->ss4o3,
         domain->ss,
         domain->v_cut,
         domain->bad_vol_h,
	 domain->cost,
	 domain->regCSR,
	 domain->regReps,
	 domain->numReg
         );

    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("ERROR: ApplyMaterialPropertiesAndUpdateVolume kernel failed: %s\n", 
               cudaGetErrorString(kernelError));
    } else {
        printf("DEBUG: ApplyMaterialPropertiesAndUpdateVolume kernel launched successfully\n");
    }
    fflush(stdout);

    //cudaDeviceSynchronize();
    //cudaCheckError();
  } else {
    printf("DEBUG: No elements to process in ApplyMaterialPropertiesAndUpdateVolume\n");
    fflush(stdout);
  }
}


// static inline
/*
 * CalcTimeConstraintsForElems - Calculate the stable timestep for the next iteration
 *
 * This function computes the maximum allowable timestep based on the Courant-Friedrichs-Lewy
 * (CFL) condition and volume change constraints. It uses two CUDA kernels:
 * 1. First kernel computes per-block minimum timesteps
 * 2. Second kernel performs global reduction to find overall minimum
 *
 * The timestep calculation is a critical part of explicit simulations to ensure
 * numerical stability. If the timestep is too large, the simulation can become unstable.
 */
 // at most 2 streams
void CalcTimeConstraintsForElems(Domain* domain, cudaStream_t* streams)
{
    printf("DEBUG: Entered CalcTimeConstraintsForElems\n");
    fflush(stdout);
    
    // Check if domain pointers are valid
    if (domain->dthydro_h == nullptr) {
        printf("ERROR: domain->dthydro_h is NULL\n");
        fflush(stdout);
    }
    if (domain->dtcourant_h == nullptr) {
        printf("ERROR: domain->dtcourant_h is NULL\n");
        fflush(stdout);
    }
    if (domain->matElemlist == nullptr) {
        printf("ERROR: domain->matElemlist is NULL\n");
        fflush(stdout);
    }
    if (domain->ss == nullptr) {
        printf("ERROR: domain->ss is NULL\n");
        fflush(stdout);
    }
    if (domain->vdov == nullptr) {
        printf("ERROR: domain->vdov is NULL\n");
        fflush(stdout);
    }
    if (domain->arealg == nullptr) {
        printf("ERROR: domain->arealg is NULL\n");
        fflush(stdout);
    }
    
    // Get artificial viscosity coefficient from domain
    Real_t qqc = domain->qqc;
    // Square and scale the coefficient (64.0 is a constant for the CFL calculation)
    Real_t qqc2 = Real_t(64.0) * qqc * qqc;
    // Maximum allowable relative volume change
    Real_t dvovmax = domain->dvovmax;

    // Total number of elements to process
    const Index_t length = domain->numElem;

    // Configure kernel launch parameters
    const int max_dimGrid = 1024;   // Maximum number of blocks
    const int dimBlock = 128;       // Threads per block
    // Calculate actual number of blocks needed, with padding
    int dimGrid = std::min(max_dimGrid, PAD_DIV(length, dimBlock));

    // Configure cache behavior for the kernel - prefer shared memory
    // This is important because the kernel uses shared memory for reduction
    cudaFuncSetCacheConfig(CalcTimeConstraintsForElems_kernel<dimBlock>, cudaFuncCachePreferShared);

    // Allocate device memory for per-block minimum timesteps
    Real_t* dev_mindtcourant;
    Real_t* dev_mindthydro;
    cudaMallocAsync((void**)&dev_mindtcourant, dimGrid * sizeof(Real_t), streams[0]);
    cudaMallocAsync((void**)&dev_mindthydro, dimGrid * sizeof(Real_t), streams[0]);

    // Launch kernel to compute per-block minimum timesteps
    // Each block processes a portion of the elements and finds local minimums
    CalcTimeConstraintsForElems_kernel<dimBlock> <<<dimGrid, dimBlock,0,streams[0]>>>
        (length, qqc2, dvovmax,
         domain->matElemlist, domain->ss, domain->vdov, domain->arealg,
         dev_mindtcourant, dev_mindthydro);

    // TODO: if dimGrid < 1024, should launch less threads
    // Launch second kernel to find global minimum across all blocks
    // This kernel performs the final reduction and stores results in domain
    CalcMinDtOneBlock<max_dimGrid> <<<2, max_dimGrid, max_dimGrid*sizeof(Real_t), streams[0]>>>
        (dev_mindthydro, dev_mindtcourant, domain->dtcourant_h, domain->dthydro_h, dimGrid);

    // Record event to track when timestep calculation is complete
    // This allows other operations to wait for this calculation to finish
    cudaEventRecord(domain->time_constraint_computed, streams[0]);

    // Free temporary device memory
    cudaFreeAsync(dev_mindtcourant, streams[0]);
    cudaFreeAsync(dev_mindthydro, streams[0]);
}

