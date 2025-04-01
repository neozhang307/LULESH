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
#include "util.h"
#include "sm_utils.inl"
#include "allocator.h"
#include "lulesh_kernels.h"
// checkErrors function is already defined at line 75






void CalcVolumeForceForElems(const Real_t hgcoef,Domain *domain)
{
    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

#ifdef DOUBLE_PRECISION
    Vector_d<Real_t>* fx_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fy_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fz_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
#else
    thrust::fill(domain->fx.begin(),domain->fx.end(),0.);
    thrust::fill(domain->fy.begin(),domain->fy.end(),0.);
    thrust::fill(domain->fz.begin(),domain->fz.end(),0.);
#endif

    int num_threads = numElem ;
    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);

    bool hourg_gt_zero = hgcoef > Real_t(0.0);
    if (hourg_gt_zero)
    {
      CalcVolumeForceForElems_kernel<true> <<<dimGrid,block_size>>>
      ( domain->volo.raw(), 
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }
    else
    {
      CalcVolumeForceForElems_kernel<false> <<<dimGrid,block_size>>>
      ( domain->volo.raw(),
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }

#ifdef DOUBLE_PRECISION
    num_threads = domain->numNode;

    // Launch boundary nodes first
    dimGrid= PAD_DIV(num_threads,block_size);

    AddNodeForcesFromElems_kernel<<<dimGrid,block_size>>>
    ( domain->numNode,
      domain->padded_numNode,
      domain->nodeElemCount.raw(),
      domain->nodeElemStart.raw(),
      domain->nodeElemCornerList.raw(),
      fx_elem->raw(),
      fy_elem->raw(),
      fz_elem->raw(),
      domain->fx.raw(),
      domain->fy.raw(),
      domain->fz.raw(),
      num_threads
    );

    Allocator<Vector_d<Real_t> >::free(fx_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fy_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fz_elem,padded_numElem*8);

#endif // ifdef DOUBLE_PRECISION
   return ;
}


// Remove static qualifier for function used across files
void CalcVolumeForceForElems(Domain* domain)
{
      const Real_t hgcoef = domain->hgcoef ;

     CalcVolumeForceForElems(hgcoef,domain);

     //CalcVolumeForceForElems_warp_per_4cell(hgcoef,domain);
}




// static inline
void CalcAccelerationForNodes(Domain *domain)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcAccelerationForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numNode,
         domain->xdd.raw(),domain->ydd.raw(),domain->zdd.raw(),
         domain->fx.raw(),domain->fy.raw(),domain->fz.raw(),
         domain->nodalMass.raw());

    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void ApplyAccelerationBoundaryConditionsForNodes(Domain *domain)
{

    Index_t dimBlock = 128;

    Index_t dimGrid = PAD_DIV(domain->numSymmX,dimBlock);
    if (domain->numSymmX > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmX,
         domain->xdd.raw(),
         domain->symmX.raw());

    dimGrid = PAD_DIV(domain->numSymmY,dimBlock);
    if (domain->numSymmY > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmY,
         domain->ydd.raw(),
         domain->symmY.raw());

    dimGrid = PAD_DIV(domain->numSymmZ,dimBlock);
    if (domain->numSymmZ > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmZ,
         domain->zdd.raw(),
         domain->symmZ.raw());
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

void CalcPositionAndVelocityForNodes(const Real_t u_cut, Domain* domain)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcPositionAndVelocityForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numNode,domain->deltatime_h,u_cut,
         domain->x.raw(),domain->y.raw(),domain->z.raw(),
         domain->xd.raw(),domain->yd.raw(),domain->zd.raw(),
         domain->xdd.raw(),domain->ydd.raw(),domain->zdd.raw());

    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void CalcKinematicsAndMonotonicQGradient(Domain *domain)
{
    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

    int num_threads = numElem;

    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);

    CalcKinematicsAndMonotonicQGradient_kernel<<<dimGrid,block_size>>>
    (  numElem,padded_numElem, domain->deltatime_h, 
       domain->nodelist.raw(),
       domain->volo.raw(),
       domain->v.raw(),
       domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
       domain->vnew->raw(),
       domain->delv.raw(),
       domain->arealg.raw(),
       domain->dxx->raw(),
       domain->dyy->raw(),
       domain->dzz->raw(),
       domain->vdov.raw(), 
       domain->delx_zeta->raw(),
       domain->delv_zeta->raw(), 
       domain->delx_xi->raw(), 
       domain->delv_xi->raw(),  
       domain->delx_eta->raw(), 
       domain->delv_eta->raw(),
       domain->bad_vol_h,
       num_threads  
    );

    //cudaDeviceSynchronize();
    //cudaCheckError();
}



void CalcMonotonicQRegionForElems(Domain *domain)
{

    const Real_t ptiny        = Real_t(1.e-36) ;
    Real_t monoq_max_slope    = domain->monoq_max_slope ;
    Real_t monoq_limiter_mult = domain->monoq_limiter_mult ;

    Real_t qlc_monoq = domain->qlc_monoq;
    Real_t qqc_monoq = domain->qqc_monoq;
    Index_t elength = domain->numElem;

    Index_t dimBlock= 128;
    Index_t dimGrid = PAD_DIV(elength,dimBlock);

    CalcMonotonicQRegionForElems_kernel<<<dimGrid,dimBlock>>>
    ( qlc_monoq,qqc_monoq,monoq_limiter_mult,monoq_max_slope,ptiny,elength,
      domain->regElemlist.raw(),domain->elemBC.raw(),
      domain->lxim.raw(),domain->lxip.raw(),
      domain->letam.raw(),domain->letap.raw(),
      domain->lzetam.raw(),domain->lzetap.raw(),
      domain->delv_xi->raw(),domain->delv_eta->raw(),domain->delv_zeta->raw(),
      domain->delx_xi->raw(),domain->delx_eta->raw(),domain->delx_zeta->raw(),
      domain->vdov.raw(),domain->elemMass.raw(),domain->volo.raw(),domain->vnew->raw(),
      domain->qq.raw(),domain->ql.raw(), 
      domain->q.raw(),
      domain->qstop,
      domain->bad_q_h
    );

    //cudaDeviceSynchronize();
    //cudaCheckError();
}






void ApplyMaterialPropertiesAndUpdateVolume(Domain *domain)
{
  Index_t length = domain->numElem ;

  if (length != 0) {

    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(length,dimBlock);

    ApplyMaterialPropertiesAndUpdateVolume_kernel<<<dimGrid,dimBlock>>>
        (length,
         domain->refdens,
         domain->e_cut,
         domain->emin,
         domain->ql.raw(),
         domain->qq.raw(),
         domain->vnew->raw(),
         domain->v.raw(),
         domain->pmin,
         domain->p_cut,
         domain->q_cut,
         domain->eosvmin,
         domain->eosvmax,
         domain->regElemlist.raw(),
         domain->e.raw(),
         domain->delv.raw(),
         domain->p.raw(),
         domain->q.raw(),
         domain->ss4o3,
         domain->ss.raw(),
         domain->v_cut,
         domain->bad_vol_h,
	 domain->cost,
	 domain->regCSR.raw(),
	 domain->regReps.raw(),
	 domain->numReg
         );

    //cudaDeviceSynchronize();
    //cudaCheckError();
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
void CalcTimeConstraintsForElems(Domain* domain)
{
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
    Vector_d<Real_t>* dev_mindtcourant = Allocator< Vector_d<Real_t> >::allocate(dimGrid);
    Vector_d<Real_t>* dev_mindthydro = Allocator< Vector_d<Real_t> >::allocate(dimGrid);

    // Launch kernel to compute per-block minimum timesteps
    // Each block processes a portion of the elements and finds local minimums
    CalcTimeConstraintsForElems_kernel<dimBlock> <<<dimGrid, dimBlock>>>
        (length, qqc2, dvovmax,
         domain->matElemlist.raw(), domain->ss.raw(), domain->vdov.raw(), domain->arealg.raw(),
         dev_mindtcourant->raw(), dev_mindthydro->raw());

    // TODO: if dimGrid < 1024, should launch less threads
    // Launch second kernel to find global minimum across all blocks
    // This kernel performs the final reduction and stores results in domain
    CalcMinDtOneBlock<max_dimGrid> <<<2, max_dimGrid, max_dimGrid*sizeof(Real_t), domain->streams[1]>>>
        (dev_mindthydro->raw(), dev_mindtcourant->raw(), domain->dtcourant_h, domain->dthydro_h, dimGrid);

    // Record event to track when timestep calculation is complete
    // This allows other operations to wait for this calculation to finish
    cudaEventRecord(domain->time_constraint_computed, domain->streams[1]);

    // Free temporary device memory
    Allocator<Vector_d<Real_t> >::free(dev_mindtcourant, dimGrid);
    Allocator<Vector_d<Real_t> >::free(dev_mindthydro, dimGrid);
}

