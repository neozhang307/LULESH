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
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>

#include <utility/util.h>
#include <utility/sm_utils.inl>
#include <cuda.h>
#include <utility/allocator.h>
#include "cuda_profiler_api.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "lulesh_split.h"

// Print usage information for command line
void printUsage(char *argv[])
{
  printf("Usage: \n");
  printf("Unstructured grid:  %s -u <file.lmesh> \n", argv[0]);
  printf("Structured grid:    %s -s numEdgeElems \n", argv[0]);
  printf("\nExamples:\n");
  printf("%s -s 45\n", argv[0]);
  printf("%s -u sedov15oct.lmesh\n", argv[0]);
}

// Initialize CUDA context for GPU
void cuda_init(int rank)
{
    Int_t deviceCount, dev;
    cudaDeviceProp cuda_deviceProp;
    
    cudaSafeCall( cudaGetDeviceCount(&deviceCount) );
    if (deviceCount == 0) {
        fprintf(stderr, "cuda_init(): no devices supporting CUDA.\n");
        exit(1);
    }

    dev = rank % deviceCount;

    if ((dev < 0) || (dev > deviceCount-1)) {
        fprintf(stderr, "cuda_init(): requested device (%d) out of range [%d,%d]\n",
                dev, 0, deviceCount-1);
        exit(1);
    }

    cudaSafeCall( cudaSetDevice(dev) );

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("Host %s using GPU %i: %s\n", hostname, dev, props.name);

    cudaSafeCall( cudaGetDeviceProperties(&cuda_deviceProp, dev) );
    if (cuda_deviceProp.major < 3) {
        fprintf(stderr, "cuda_init(): This implementation of Lulesh requires device SM 3.0+.\n", dev);
        exit(1);
    }

#if CUDART_VERSION < 5000
   fprintf(stderr,"cuda_init(): This implementation of Lulesh uses texture objects, which is requires Cuda 5.0+.\n");
   exit(1);
#endif

}
void AllocateNodalPersistent(Domain* domain, size_t domNodes)
{
  domain->x.resize(domNodes) ;  /* coordinates */
  domain->y.resize(domNodes) ;
  domain->z.resize(domNodes) ;

  domain->xd.resize(domNodes) ; /* velocities */
  domain->yd.resize(domNodes) ;
  domain->zd.resize(domNodes) ;

  domain->xdd.resize(domNodes) ; /* accelerations */
  domain->ydd.resize(domNodes) ;
  domain->zdd.resize(domNodes) ;

  domain->fx.resize(domNodes) ;  /* forces */
  domain->fy.resize(domNodes) ;
  domain->fz.resize(domNodes) ;

  domain->nodalMass.resize(domNodes) ;  /* mass */
}

void AllocateElemPersistent(Domain* domain, size_t domElems, size_t padded_domElems)
{
   domain->matElemlist.resize(domElems) ;  /* material indexset */
   domain->nodelist.resize(8*padded_domElems) ;   /* elemToNode connectivity */

   domain->lxim.resize(domElems) ; /* elem connectivity through face */
   domain->lxip.resize(domElems) ;
   domain->letam.resize(domElems) ;
   domain->letap.resize(domElems) ;
   domain->lzetam.resize(domElems) ;
   domain->lzetap.resize(domElems) ;

   domain->elemBC.resize(domElems) ;  /* elem face symm/free-surf flag */

   domain->e.resize(domElems) ;   /* energy */
   domain->p.resize(domElems) ;   /* pressure */

   domain->q.resize(domElems) ;   /* q */
   domain->ql.resize(domElems) ;  /* linear term for q */
   domain->qq.resize(domElems) ;  /* quadratic term for q */

   domain->v.resize(domElems) ;     /* relative volume */

   domain->volo.resize(domElems) ;  /* reference volume */
   domain->delv.resize(domElems) ;  /* m_vnew - m_v */
   domain->vdov.resize(domElems) ;  /* volume derivative over volume */

   domain->arealg.resize(domElems) ;  /* elem characteristic length */

   domain->ss.resize(domElems) ;      /* "sound speed" */

   domain->elemMass.resize(domElems) ;  /* mass */

}

void AllocateSymmX(Domain* domain, size_t size)
{
   domain->symmX.resize(size) ;
}

void AllocateSymmY(Domain* domain, size_t size)
{
   domain->symmY.resize(size) ;
}

void AllocateSymmZ(Domain* domain, size_t size)
{
   domain->symmZ.resize(size) ;
}

void InitializeFields(Domain* domain)
{
 /* Basic Field Initialization */

 thrust::fill(domain->ss.begin(),domain->ss.end(),0.);
 thrust::fill(domain->e.begin(),domain->e.end(),0.);
 thrust::fill(domain->p.begin(),domain->p.end(),0.);
 thrust::fill(domain->q.begin(),domain->q.end(),0.);
 thrust::fill(domain->v.begin(),domain->v.end(),1.);

 thrust::fill(domain->xd.begin(),domain->xd.end(),0.);
 thrust::fill(domain->yd.begin(),domain->yd.end(),0.);
 thrust::fill(domain->zd.begin(),domain->zd.end(),0.);

 thrust::fill(domain->xdd.begin(),domain->xdd.end(),0.);
 thrust::fill(domain->ydd.begin(),domain->ydd.end(),0.);
 thrust::fill(domain->zdd.begin(),domain->zdd.end(),0.);

 thrust::fill(domain->nodalMass.begin(),domain->nodalMass.end(),0.);
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // allocate a buffer large enough for nodal ghost data 
  maxEdgeSize = MAX(this->sizeX, MAX(this->sizeY, this->sizeZ))+1 ;
  maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

  // assume communication to 6 neighbors by default 
  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

#if USE_MPI   
  // account for face communication 
  Index_t comBufSize =
    (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
    maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for edge communication 
  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for corner communication 
  // factor of 16 is so each buffer has its own cache line 
  comBufSize += ((m_rowMin & m_colMin & m_planeMin) +
                 (m_rowMin & m_colMin & m_planeMax) +
                 (m_rowMin & m_colMax & m_planeMin) +
                 (m_rowMin & m_colMax & m_planeMax) +
                 (m_rowMax & m_colMin & m_planeMin) +
                 (m_rowMax & m_colMin & m_planeMax) +
                 (m_rowMax & m_colMax & m_planeMin) +
                 (m_rowMax & m_colMax & m_planeMax)) * CACHE_COHERENCE_PAD_REAL ;

  this->commDataSend = new Real_t[comBufSize] ;
  this->commDataRecv = new Real_t[comBufSize] ;

  // pin buffers
  cudaHostRegister(this->commDataSend, comBufSize*sizeof(Real_t), 0);
  cudaHostRegister(this->commDataRecv, comBufSize*sizeof(Real_t), 0);

  // prevent floating point exceptions 
  memset(this->commDataSend, 0, comBufSize*sizeof(Real_t)) ;
  memset(this->commDataRecv, 0, comBufSize*sizeof(Real_t)) ;

  // allocate shadow GPU buffers
  cudaMalloc(&this->d_commDataSend, comBufSize*sizeof(Real_t));
  cudaMalloc(&this->d_commDataRecv, comBufSize*sizeof(Real_t));
  
  // prevent floating point exceptions 
  cudaMemset(this->d_commDataSend, 0, comBufSize*sizeof(Real_t));
  cudaMemset(this->d_commDataRecv, 0, comBufSize*sizeof(Real_t));
#endif
}

void SetupConnectivityBC(Domain *domain, int edgeElems)
{
  int domElems = domain->numElem;

  Vector_h<Index_t> lxim_h(domElems);
  Vector_h<Index_t> lxip_h(domElems);
  Vector_h<Index_t> letam_h(domElems);
  Vector_h<Index_t> letap_h(domElems);
  Vector_h<Index_t> lzetam_h(domElems);
  Vector_h<Index_t> lzetap_h(domElems);

    /* set up elemement connectivity information */
    lxim_h[0] = 0 ;
    for (Index_t i=1; i<domElems; ++i) {
       lxim_h[i]   = i-1 ;
       lxip_h[i-1] = i ;
    }
    lxip_h[domElems-1] = domElems-1 ;

    for (Index_t i=0; i<edgeElems; ++i) {
       letam_h[i] = i ;
       letap_h[domElems-edgeElems+i] = domElems-edgeElems+i ;
    }
    for (Index_t i=edgeElems; i<domElems; ++i) {
       letam_h[i] = i-edgeElems ;
       letap_h[i-edgeElems] = i ;
    }

    for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
       lzetam_h[i] = i ;
       lzetap_h[domElems-edgeElems*edgeElems+i] = domElems-edgeElems*edgeElems+i ;
    }
    for (Index_t i=edgeElems*edgeElems; i<domElems; ++i) {
       lzetam_h[i] = i - edgeElems*edgeElems ;
       lzetap_h[i-edgeElems*edgeElems] = i ;
    }


  /* set up boundary condition information */
  Vector_h<Index_t> elemBC_h(domElems);
  for (Index_t i=0; i<domElems; ++i) {
     elemBC_h[i] = 0 ;  /* clear BCs by default */
  }

  Index_t ghostIdx[6] ;  // offsets to ghost locations

  for (Index_t i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = domElems ;
  if (domain->m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += domain->sizeX*domain->sizeY ;
  }

  if (domain->m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += domain->sizeX*domain->sizeY ;
  }

  if (domain->m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += domain->sizeX*domain->sizeZ ;
  }

  if (domain->m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += domain->sizeX*domain->sizeZ ;
  }

  if (domain->m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += domain->sizeY*domain->sizeZ ;
  }

  if (domain->m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  /* symmetry plane or free surface BCs */
  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (domain->m_planeLoc == 0) {
        elemBC_h[rowInc+j] |= ZETA_M_SYMM ;
      }
      else {
        elemBC_h[rowInc+j] |= ZETA_M_COMM ;
        lzetam_h[rowInc+j] = ghostIdx[0] + rowInc + j ;
      }

      if (domain->m_planeLoc == domain->m_tp-1) {
        elemBC_h[rowInc+j+domElems-edgeElems*edgeElems] |=
          ZETA_P_FREE;
      }
      else {
        elemBC_h[rowInc+j+domElems-edgeElems*edgeElems] |=
          ZETA_P_COMM ;
        lzetap_h[rowInc+j+domElems-edgeElems*edgeElems] =
          ghostIdx[1] + rowInc + j ;
      }

      if (domain->m_rowLoc == 0) {
        elemBC_h[planeInc+j] |= ETA_M_SYMM ;
      }
      else {
        elemBC_h[planeInc+j] |= ETA_M_COMM ;
        letam_h[planeInc+j] = ghostIdx[2] + rowInc + j ;
      }

      if (domain->m_rowLoc == domain->m_tp-1) {
        elemBC_h[planeInc+j+edgeElems*edgeElems-edgeElems] |=
          ETA_P_FREE ;
      }
      else {
        elemBC_h[planeInc+j+edgeElems*edgeElems-edgeElems] |=
          ETA_P_COMM ;
        letap_h[planeInc+j+edgeElems*edgeElems-edgeElems] =
          ghostIdx[3] +  rowInc + j ;
      }

      if (domain->m_colLoc == 0) {
        elemBC_h[planeInc+j*edgeElems] |= XI_M_SYMM ;
      }
      else {
        elemBC_h[planeInc+j*edgeElems] |= XI_M_COMM ;
        lxim_h[planeInc+j*edgeElems] = ghostIdx[4] + rowInc + j ;
      }

      if (domain->m_colLoc == domain->m_tp-1) {
        elemBC_h[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
      }
      else {
        elemBC_h[planeInc+j*edgeElems+edgeElems-1] |= XI_P_COMM ;
        lxip_h[planeInc+j*edgeElems+edgeElems-1] =
          ghostIdx[5] + rowInc + j ;
      }
    }
  }

  domain->elemBC = elemBC_h;
  domain->lxim = lxim_h;
  domain->lxip = lxip_h;
  domain->letam = letam_h;
  domain->letap = letap_h;
  domain->lzetam = lzetam_h;
  domain->lzetap = lzetap_h;
}

void Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Int_t domNodes, Int_t padded_domElems, Vector_h<Real_t> &x_h, Vector_h<Real_t> &y_h, Vector_h<Real_t> &z_h, Vector_h<Int_t> &nodelist_h)
{
  Index_t meshEdgeElems = m_tp*nx ;

  x_h.resize(domNodes);
  y_h.resize(domNodes);
  z_h.resize(domNodes);

  // initialize nodal coordinates 
  Index_t nidx = 0 ;
  Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
  for (Index_t plane=0; plane<edgeNodes; ++plane) {
    Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
    for (Index_t row=0; row<edgeNodes; ++row) {
      Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
      for (Index_t col=0; col<edgeNodes; ++col) {
        x_h[nidx] = tx ;
        y_h[nidx] = ty ;
        z_h[nidx] = tz ;
        ++nidx ;
        // tx += ds ; // may accumulate roundoff... 
        tx = Real_t(1.125)*Real_t(m_colLoc*nx+col+1)/Real_t(meshEdgeElems) ;
      }
      // ty += ds ;  // may accumulate roundoff... 
      ty = Real_t(1.125)*Real_t(m_rowLoc*nx+row+1)/Real_t(meshEdgeElems) ;
    }
    // tz += ds ;  // may accumulate roundoff... 
    tz = Real_t(1.125)*Real_t(m_planeLoc*nx+plane+1)/Real_t(meshEdgeElems) ;
  }

  x = x_h;
  y = y_h;
  z = z_h;

  nodelist_h.resize(padded_domElems*8);

  // embed hexehedral elements in nodal point lattice 
  Index_t zidx = 0 ;
  nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
        nodelist_h[0*padded_domElems+zidx] = nidx                                       ;
        nodelist_h[1*padded_domElems+zidx] = nidx                                   + 1 ;
        nodelist_h[2*padded_domElems+zidx] = nidx                       + edgeNodes + 1 ;
        nodelist_h[3*padded_domElems+zidx] = nidx                       + edgeNodes     ;
        nodelist_h[4*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes                 ;
        nodelist_h[5*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes             + 1 ;
        nodelist_h[6*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
        nodelist_h[7*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
        ++zidx ;
        ++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }

  nodelist = nodelist_h;
}
void Domain::CreateRegionIndexSets(Int_t nr, Int_t b)
{
#if USE_MPI   
   Index_t myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   srand(myRank);
#else
   srand(0);
   Index_t myRank = 0;
#endif
   numReg = nr;
   balance = b;

   regElemSize = new Int_t[numReg];
   Index_t nextIndex = 0;

   Vector_h<Int_t> regCSR_h(regCSR.size());  // records the begining and end of each region
   Vector_h<Int_t> regReps_h(regReps.size()); // records the rep number per region
   Vector_h<Index_t> regNumList_h(regNumList.size());    // Region number per domain element
   Vector_h<Index_t> regElemlist_h(regElemlist.size());  // region indexset 
   Vector_h<Index_t> regSorted_h(regSorted.size()); // keeps index of sorted regions

   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(numReg == 1) {
      while (nextIndex < numElem) {
         regNumList_h[nextIndex] = 1;
         nextIndex++;
      }
      regElemSize[0] = 0;
   }
   //If we have more than one region distribute the elements.
   else {
      Int_t regionNum;
      Int_t regionVar;
      Int_t lastReg = -1;
      Int_t binSize;
      Int_t elements;
      Index_t runto = 0;
      Int_t costDenominator = 0;
      Int_t* regBinEnd = new Int_t[numReg];
      //Determine the relative weights of all the regions.
      for (Index_t i=0 ; i<numReg ; ++i) {
         regElemSize[i] = 0;
         costDenominator += POW((i+1), balance);  //Total cost of all regions
         regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < numElem) {
         //pick the region
         regionVar = rand() % costDenominator;
         Index_t i = 0;
         while(regionVar >= regBinEnd[i])
            i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
         regionNum = ((i + myRank) % numReg) + 1;
         // make sure we don't pick the same region twice in a row
         while(regionNum == lastReg) {
            regionVar = rand() % costDenominator;
            i = 0;
            while(regionVar >= regBinEnd[i])
               i++;
            regionNum = ((i + myRank) % numReg) + 1;
         }
         //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if(binSize < 773) {
           elements = rand() % 15 + 1;
         }
         else if(binSize < 937) {
           elements = rand() % 16 + 16;
         }
         else if(binSize < 970) {
           elements = rand() % 32 + 32;
         }
         else if(binSize < 974) {
           elements = rand() % 64 + 64;
         }
         else if(binSize < 978) {
           elements = rand() % 128 + 128;
         }
         else if(binSize < 981) {
           elements = rand() % 256 + 256;
         }
         else
            elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
         //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < numElem) {
            regNumList_h[nextIndex] = regionNum;
            nextIndex++;
         }
         lastReg = regionNum;
      }
   }
   // Convert regNumList to region index sets
   // First, count size of each region 
   for (Index_t i=0 ; i<numElem ; ++i) {
      int r = regNumList_h[i]-1; // region index == regnum-1
      regElemSize[r]++;
   }

   Index_t rep;
   // Second, allocate each region index set
   for (Index_t r=0; r<numReg ; ++r) {
       if(r < numReg/2)
         rep = 1;
       else if(r < (numReg - (numReg+15)/20))
         rep = 1 + cost;
       else
         rep = 10 * (1+ cost);
       regReps_h[r] = rep;
   }

   sortRegions(regReps_h, regSorted_h);

   regCSR_h[0] = 0;
   // Second, allocate each region index set
   for (Index_t i=1 ; i<numReg ; ++i) {
      regCSR_h[i] = regCSR_h[i-1] + regElemSize[i-1];
   }

   // Third, fill index sets
   for (Index_t i=0 ; i<numElem ; ++i) {
      Index_t r = regSorted_h[regNumList_h[i]-1];       // region index == regnum-1
      regElemlist_h[regCSR_h[r]] = i;
      regCSR_h[r]++;
   }

   // Copy to device
   regCSR =  regCSR_h;  // records the begining and end of each region
   regReps =  regReps_h; // records the rep number per region
   regNumList =  regNumList_h;    // Region number per domain element
   regElemlist = regElemlist_h;  // region indexset 
   regSorted = regSorted_h; // keeps index of sorted regions

} // end of create function
void Domain::sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h)
{
  Index_t temp;
  Vector_h<Index_t> regIndex;
  regIndex.resize(numReg);
  for(int i = 0; i < numReg; i++)
	regIndex[i] = i;

  for(int i = 0; i < numReg-1; i++)
	for(int j = 0; j < numReg-i-1; j++)
		if(regReps_h[j] < regReps_h[j+1])
		{
                  temp = regReps_h[j];
                  regReps_h[j] = regReps_h[j+1];
                  regReps_h[j+1] = temp;

		  temp = regElemSize[j];
		  regElemSize[j] = regElemSize[j+1];
		  regElemSize[j+1] = temp;

                  temp = regIndex[j];
                  regIndex[j] = regIndex[j+1];
                  regIndex[j+1] = temp;
		}
  for(int i = 0; i < numReg; i++)
        regSorted_h[regIndex[i]] = i;
}
///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;
   
   // Assume cube processor layout for now 
   testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("MPI operations only support float and double right now...\n");
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   // temporary test
   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   Int_t remainder = dx*dy*dz % numRanks ;
   if (myRank < remainder) {
      myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
   }
   else {
      myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
         (myRank - remainder)*(dx*dy*dz/numRanks) ;
   }

   *col = myDom % dx ;
   *row = (myDom / dx) % dy ;
   *plane = myDom / (dx*dy) ;
   *side = testProcs;

   return;
}


Domain *NewDomain(char* argv[], Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, bool structured, Int_t nr, Int_t balance, Int_t cost)
{

  Domain *domain = new Domain ;

  domain->max_streams = 32;
  domain->streams.resize(domain->max_streams);

  for (Int_t i=0;i<domain->max_streams;i++)
    cudaStreamCreate(&(domain->streams[i]));

  cudaEventCreateWithFlags(&domain->time_constraint_computed,cudaEventDisableTiming);

  Index_t domElems;
  Index_t domNodes;
  Index_t padded_domElems;

  Vector_h<Index_t> nodelist_h;
  Vector_h<Real_t> x_h;
  Vector_h<Real_t> y_h;
  Vector_h<Real_t> z_h;

  if (structured)
  {
    domain->m_tp       = tp ;
    domain->m_numRanks = numRanks ;

    domain->m_colLoc   =   colLoc ;
    domain->m_rowLoc   =   rowLoc ;
    domain->m_planeLoc = planeLoc ;

    Index_t edgeElems = nx ;
    Index_t edgeNodes = edgeElems+1 ;

    domain->sizeX = edgeElems ;
    domain->sizeY = edgeElems ;
    domain->sizeZ = edgeElems ;  

    domain->numElem = domain->sizeX*domain->sizeY*domain->sizeZ ;
    domain->padded_numElem = PAD(domain->numElem,32);

    domain->numNode = (domain->sizeX+1)*(domain->sizeY+1)*(domain->sizeZ+1) ;
    domain->padded_numNode = PAD(domain->numNode,32);

    domElems = domain->numElem ;
    domNodes = domain->numNode ;
    padded_domElems = domain->padded_numElem ;

    AllocateElemPersistent(domain,domElems,padded_domElems);
    AllocateNodalPersistent(domain,domNodes);

    domain->SetupCommBuffers(edgeNodes);

    InitializeFields(domain);

    domain->BuildMesh(nx, edgeNodes, edgeElems, domNodes, padded_domElems, x_h, y_h, z_h, nodelist_h);

    domain->numSymmX = domain->numSymmY = domain->numSymmZ = 0;

    if (domain->m_colLoc == 0) 
      domain->numSymmX = (edgeElems+1)*(edgeElems+1) ;
    if (domain->m_rowLoc == 0) 
      domain->numSymmY = (edgeElems+1)*(edgeElems+1) ;
    if (domain->m_planeLoc == 0)
      domain->numSymmZ = (edgeElems+1)*(edgeElems+1) ;

    AllocateSymmX(domain,edgeNodes*edgeNodes);
    AllocateSymmY(domain,edgeNodes*edgeNodes);
    AllocateSymmZ(domain,edgeNodes*edgeNodes);

    /* set up symmetry nodesets */

    Vector_h<Index_t> symmX_h(domain->symmX.size());
    Vector_h<Index_t> symmY_h(domain->symmY.size());
    Vector_h<Index_t> symmZ_h(domain->symmZ.size());

    Int_t nidx = 0 ;
    for (Index_t i=0; i<edgeNodes; ++i) {
       Index_t planeInc = i*edgeNodes*edgeNodes ;
       Index_t rowInc   = i*edgeNodes ;
       for (Index_t j=0; j<edgeNodes; ++j) {
         if (domain->m_planeLoc == 0) {
           symmZ_h[nidx] = rowInc   + j ;
         } 
         if (domain->m_rowLoc == 0) {
           symmY_h[nidx] = planeInc + j ;
         }
         if (domain->m_colLoc == 0) {
           symmX_h[nidx] = planeInc + j*edgeNodes ;
         }
        ++nidx ;
       }
    }

    if (domain->m_planeLoc == 0)
      domain->symmZ = symmZ_h;
    if (domain->m_rowLoc == 0)
      domain->symmY = symmY_h;
    if (domain->m_colLoc == 0)
      domain->symmX = symmX_h;

    SetupConnectivityBC(domain, edgeElems);
  }
  else
  {
    FILE *fp;
    int ee, en;

    if ((fp = fopen(argv[2], "r")) == 0) {
       printf("could not open file %s\n", argv[2]) ;
       exit( LFileError ) ;
    }

    bool fsuccess;
    fsuccess = fscanf(fp, "%d %d", &ee, &en) ;
    domain->numElem = Index_t(ee);
    domain->padded_numElem = PAD(domain->numElem,32);

    domain->numNode = Index_t(en);
    domain->padded_numNode = PAD(domain->numNode,32);

    domElems = domain->numElem ;
    domNodes = domain->numNode ;
    padded_domElems = domain->padded_numElem ;

    AllocateElemPersistent(domain,domElems,padded_domElems);
    AllocateNodalPersistent(domain,domNodes);

    InitializeFields(domain);

    /* initialize nodal coordinates */
    x_h.resize(domNodes);
    y_h.resize(domNodes);
    z_h.resize(domNodes);

    for (Index_t i=0; i<domNodes; ++i) {
       double px, py, pz ;
       fsuccess = fscanf(fp, "%lf %lf %lf", &px, &py, &pz) ;
       x_h[i] = Real_t(px) ;
       y_h[i] = Real_t(py) ;
       z_h[i] = Real_t(pz) ;
    }
    domain->x = x_h;
    domain->y = y_h;
    domain->z = z_h;

    /* embed hexehedral elements in nodal point lattice */
    nodelist_h.resize(padded_domElems*8);
    for (Index_t zidx=0; zidx<domElems; ++zidx) {
       for (Index_t ni=0; ni<Index_t(8); ++ni) {
          int n ;
          fsuccess = fscanf(fp, "%d", &n) ;
          nodelist_h[ni*padded_domElems+zidx] = Index_t(n);
       }
    }
    domain->nodelist = nodelist_h;

    /* set up face-based element neighbors */
    Vector_h<Index_t> lxim_h(domElems);
    Vector_h<Index_t> lxip_h(domElems);
    Vector_h<Index_t> letam_h(domElems);
    Vector_h<Index_t> letap_h(domElems);
    Vector_h<Index_t> lzetam_h(domElems);
    Vector_h<Index_t> lzetap_h(domElems);

    for (Index_t i=0; i<domElems; ++i) {
       int xi_m, xi_p, eta_m, eta_p, zeta_m, zeta_p ;
       fsuccess = fscanf(fp, "%d %d %d %d %d %d",
             &xi_m, &xi_p, &eta_m, &eta_p, &zeta_m, &zeta_p) ;

       lxim_h[i]   = Index_t(xi_m) ;
       lxip_h[i]   = Index_t(xi_p) ;
       letam_h[i]  = Index_t(eta_m) ;
       letap_h[i]  = Index_t(eta_p) ;
       lzetam_h[i] = Index_t(zeta_m) ;
       lzetap_h[i] = Index_t(zeta_p) ;
    }

    domain->lxim = lxim_h;
    domain->lxip = lxip_h;
    domain->letam = letam_h;
    domain->letap = letap_h;
    domain->lzetam = lzetam_h;
    domain->lzetap = lzetap_h;

    /* set up X symmetry nodeset */

    fsuccess = fscanf(fp, "%d", &domain->numSymmX) ;
    Vector_h<Index_t> symmX_h(domain->numSymmX);
    for (Index_t i=0; i<domain->numSymmX; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmX_h[i] = Index_t(n) ;
    }
    domain->symmX = symmX_h;

    fsuccess = fscanf(fp, "%d", &domain->numSymmY) ;
    Vector_h<Index_t> symmY_h(domain->numSymmY);
    for (Index_t i=0; i<domain->numSymmY; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmY_h[i] = Index_t(n) ;
    }
    domain->symmY = symmY_h;

    fsuccess = fscanf(fp, "%d", &domain->numSymmZ) ;
    Vector_h<Index_t> symmZ_h(domain->numSymmZ);
    for (Index_t i=0; i<domain->numSymmZ; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmZ_h[i] = Index_t(n) ;
    }
    domain->symmZ = symmZ_h;

    /* set up free surface nodeset */
    Index_t numFreeSurf;
    fsuccess = fscanf(fp, "%d", &numFreeSurf) ;
    Vector_h<Index_t> freeSurf_h(numFreeSurf);
    for (Index_t i=0; i<numFreeSurf; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       freeSurf_h[i] = Index_t(n) ;
    }
    printf("%c\n",fsuccess);//nothing
    fclose(fp);

    /* set up boundary condition information */
    Vector_h<Index_t> elemBC_h(domElems);
    Vector_h<Index_t> surfaceNode_h(domNodes);

    for (Index_t i=0; i<domain->numElem; ++i) {
       elemBC_h[i] = 0 ;
    }

    for (Index_t i=0; i<domain->numNode; ++i) {
       surfaceNode_h[i] = 0 ;
    }

    for (Index_t i=0; i<domain->numSymmX; ++i) {
       surfaceNode_h[symmX_h[i]] = 1 ;
    }

    for (Index_t i=0; i<domain->numSymmY; ++i) {
       surfaceNode_h[symmY_h[i]] = 1 ;
    }

    for (Index_t i=0; i<domain->numSymmZ; ++i) {
       surfaceNode_h[symmZ_h[i]] = 1 ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       Int_t mask = 0 ;

       for (Index_t ni=0; ni<8; ++ni) {
          mask |= (surfaceNode_h[nodelist_h[ni*domain->padded_numElem+zidx]] << ni) ;
       }

      if ((mask & 0x0f) == 0x0f) elemBC_h[zidx] |= ZETA_M_SYMM ;
      if ((mask & 0xf0) == 0xf0) elemBC_h[zidx] |= ZETA_P_SYMM ;
      if ((mask & 0x33) == 0x33) elemBC_h[zidx] |= ETA_M_SYMM ;
      if ((mask & 0xcc) == 0xcc) elemBC_h[zidx] |= ETA_P_SYMM ;
      if ((mask & 0x99) == 0x99) elemBC_h[zidx] |= XI_M_SYMM ;
      if ((mask & 0x66) == 0x66) elemBC_h[zidx] |= XI_P_SYMM ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       if (elemBC_h[zidx] == (XI_M_SYMM | ETA_M_SYMM | ZETA_M_SYMM)) {
          domain->octantCorner = zidx ;
          break ;
       }
    }

    for (Index_t i=0; i<domain->numNode; ++i) {
       surfaceNode_h[i] = 0 ;
    }

    for (Index_t i=0; i<numFreeSurf; ++i) {
       surfaceNode_h[freeSurf_h[i]] = 1 ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       Int_t mask = 0 ;

       for (Index_t ni=0; ni<8; ++ni) {
          mask |= (surfaceNode_h[nodelist_h[ni*domain->padded_numElem+zidx]] << ni) ;
       }

      if ((mask & 0x0f) == 0x0f) elemBC_h[zidx] |= ZETA_M_SYMM ;
      if ((mask & 0xf0) == 0xf0) elemBC_h[zidx] |= ZETA_P_SYMM ;
      if ((mask & 0x33) == 0x33) elemBC_h[zidx] |= ETA_M_SYMM ;
      if ((mask & 0xcc) == 0xcc) elemBC_h[zidx] |= ETA_P_SYMM ;
      if ((mask & 0x99) == 0x99) elemBC_h[zidx] |= XI_M_SYMM ;
      if ((mask & 0x66) == 0x66) elemBC_h[zidx] |= XI_P_SYMM ;
    }

    domain->elemBC = elemBC_h;

    /* deposit energy */
    domain->e[domain->octantCorner] = Real_t(3.948746e+7) ;

  }

  /* set up node-centered indexing of elements */
  Vector_h<Index_t> nodeElemCount_h(domNodes);

  for (Index_t i=0; i<domNodes; ++i) {
     nodeElemCount_h[i] = 0 ;
  }

  for (Index_t i=0; i<domElems; ++i) {
     for (Index_t j=0; j < 8; ++j) {
        ++(nodeElemCount_h[nodelist_h[j*padded_domElems+i]]);
     }
  }

  Vector_h<Index_t> nodeElemStart_h(domNodes);

  nodeElemStart_h[0] = 0;
  for (Index_t i=1; i < domNodes; ++i) {
     nodeElemStart_h[i] =
        nodeElemStart_h[i-1] + nodeElemCount_h[i-1] ;
  }
  
  Vector_h<Index_t> nodeElemCornerList_h(nodeElemStart_h[domNodes-1] +
                 nodeElemCount_h[domNodes-1] );

  for (Index_t i=0; i < domNodes; ++i) {
     nodeElemCount_h[i] = 0;
  }

  for (Index_t j=0; j < 8; ++j) {
    for (Index_t i=0; i < domElems; ++i) {
        Index_t m = nodelist_h[padded_domElems*j+i];
        Index_t k = padded_domElems*j + i ;
        Index_t offset = nodeElemStart_h[m] +
                         nodeElemCount_h[m] ;
        nodeElemCornerList_h[offset] = k;
        ++(nodeElemCount_h[m]) ;
     }
  }

  Index_t clSize = nodeElemStart_h[domNodes-1] +
                   nodeElemCount_h[domNodes-1] ;
  for (Index_t i=0; i < clSize; ++i) {
     Index_t clv = nodeElemCornerList_h[i] ;
     if ((clv < 0) || (clv > padded_domElems*8)) {
          fprintf(stderr,
   "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
          exit(1);
     }
  }

  domain->nodeElemStart = nodeElemStart_h;
  domain->nodeElemCount = nodeElemCount_h;
  domain->nodeElemCornerList = nodeElemCornerList_h;

  /* Create a material IndexSet (entire domain same material for now) */
  Vector_h<Index_t> matElemlist_h(domElems);
  for (Index_t i=0; i<domElems; ++i) {
     matElemlist_h[i] = i ;
  }
  domain->matElemlist = matElemlist_h;

  cudaMallocHost(&domain->dtcourant_h,sizeof(Real_t),0);
  cudaMallocHost(&domain->dthydro_h,sizeof(Real_t),0);
  cudaMallocHost(&domain->bad_vol_h,sizeof(Index_t),0);
  cudaMallocHost(&domain->bad_q_h,sizeof(Index_t),0);
 
  *(domain->bad_vol_h)=-1;
  *(domain->bad_q_h)=-1;
  *(domain->dthydro_h)=1e20;
  *(domain->dtcourant_h)=1e20;

  /* initialize material parameters */
  domain->time_h      = Real_t(0.) ;
  domain->dtfixed = Real_t(-1.0e-6) ;
  domain->deltatimemultlb = Real_t(1.1) ;
  domain->deltatimemultub = Real_t(1.2) ;
  domain->stoptime  = Real_t(1.0e-2) ;
  domain->dtmax     = Real_t(1.0e-2) ;
  domain->cycle   = 0 ;

  domain->e_cut = Real_t(1.0e-7) ;
  domain->p_cut = Real_t(1.0e-7) ;
  domain->q_cut = Real_t(1.0e-7) ;
  domain->u_cut = Real_t(1.0e-7) ;
  domain->v_cut = Real_t(1.0e-10) ;

  domain->hgcoef      = Real_t(3.0) ;
  domain->ss4o3       = Real_t(4.0)/Real_t(3.0) ;

  domain->qstop              =  Real_t(1.0e+12) ;
  domain->monoq_max_slope    =  Real_t(1.0) ;
  domain->monoq_limiter_mult =  Real_t(2.0) ;
  domain->qlc_monoq          = Real_t(0.5) ;
  domain->qqc_monoq          = Real_t(2.0)/Real_t(3.0) ;
  domain->qqc                = Real_t(2.0) ;

  domain->pmin =  Real_t(0.) ;
  domain->emin = Real_t(-1.0e+15) ;

  domain->dvovmax =  Real_t(0.1) ;

  domain->eosvmax =  Real_t(1.0e+9) ;
  domain->eosvmin =  Real_t(1.0e-9) ;

  domain->refdens =  Real_t(1.0) ;

  /* initialize field data */
  Vector_h<Real_t> nodalMass_h(domNodes);
  Vector_h<Real_t> volo_h(domElems);
  Vector_h<Real_t> elemMass_h(domElems);

  for (Index_t i=0; i<domElems; ++i) {
     Real_t x_local[8], y_local[8], z_local[8] ;
     for( Index_t lnode=0 ; lnode<8 ; ++lnode )
     {
       Index_t gnode = nodelist_h[lnode*padded_domElems+i];
       x_local[lnode] = x_h[gnode];
       y_local[lnode] = y_h[gnode];
       z_local[lnode] = z_h[gnode];
     }

     // volume calculations
     Real_t volume = CalcElemVolume(x_local, y_local, z_local );
     volo_h[i] = volume ;
     elemMass_h[i] = volume ;
     for (Index_t j=0; j<8; ++j) {
        Index_t gnode = nodelist_h[j*padded_domElems+i];
        nodalMass_h[gnode] += volume / Real_t(8.0) ;
     }
  }

  domain->nodalMass = nodalMass_h;
  domain->volo = volo_h;
  domain->elemMass= elemMass_h;

   /* deposit energy */
   domain->octantCorner = 0;
  // deposit initial energy
  // An energy of 3.948746e+7 is correct for a problem with
  // 45 zones along a side - we need to scale it
  const Real_t ebase = 3.948746e+7;
  Real_t scale = (nx*domain->m_tp)/45.0;
  Real_t einit = ebase*scale*scale*scale;
  //Real_t einit = ebase;
  if (domain->m_rowLoc + domain->m_colLoc + domain->m_planeLoc == 0) {
     // Dump into the first zone (which we know is in the corner)
     // of the domain that sits at the origin
       domain->e[0] = einit;
  }

  //set initial deltatime base on analytic CFL calculation
  domain->deltatime_h = (.5*cbrt(domain->volo[0]))/sqrt(2*einit);

  domain->cost = cost;
  domain->regNumList.resize(domain->numElem) ;  // material indexset
  domain->regElemlist.resize(domain->numElem) ;  // material indexset
  domain->regCSR.resize(nr);
  domain->regReps.resize(nr);
  domain->regSorted.resize(nr);

  // Setup region index sets. For now, these are constant sized
  // throughout the run, but could be changed every cycle to 
  // simulate effects of ALE on the lagrange solver

  domain->CreateRegionIndexSets(nr, balance);

  return domain ;
}

void write_solution(Domain* locDom)
{
  Vector_h<Real_t> x_h = locDom->x;
  Vector_h<Real_t> y_h = locDom->y;
  Vector_h<Real_t> z_h = locDom->z;

//  printf("Writing solution to file xyz.asc\n");
  std::stringstream filename;
  filename << "xyz.asc";

  FILE *fout = fopen(filename.str().c_str(),"wb");

  for (Index_t i=0; i<locDom->numNode; i++) {
      fprintf(fout,"%10d\n",i);
      fprintf(fout,"%.10f\n",x_h[i]);
      fprintf(fout,"%.10f\n",y_h[i]);
      fprintf(fout,"%.10f\n",z_h[i]);
  }
  fclose(fout);
}
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
			       Int_t its,
                               Int_t nx,
                               Int_t numRanks, 
                               bool structured)
{
   size_t free_mem, total_mem, used_mem;
   cudaMemGetInfo(&free_mem, &total_mem);
   used_mem= total_mem - free_mem;
#if LULESH_SHOW_PROGRESS == 0
   printf("   Used Memory         =  %8.4f Mb\n", used_mem / (1024.*1024.) );
#endif

   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism 
   Real_t grindTime1; 
   Real_t grindTime2;
   if(structured)
   {
      grindTime1 = ((elapsed_time*1e6)/its)/(nx*nx*nx);
      grindTime2 = ((elapsed_time*1e6)/its)/(nx*nx*nx*numRanks);
   }
   else
   {
      grindTime1 = ((elapsed_time*1e6)/its)/(locDom.numElem);
      grindTime2 = ((elapsed_time*1e6)/its)/(locDom.numElem*numRanks);
   }
   // Copy Energy back to Host 
   if(structured)
   {
      Real_t e_zero;
      Real_t* d_ezero_ptr = locDom.e.raw() + locDom.octantCorner; /* octant corner supposed to be 0 */
      cudaMemcpy(&e_zero, d_ezero_ptr, sizeof(Real_t), cudaMemcpyDeviceToHost);

      printf("Run completed:  \n");
      printf("   Problem size        =  %i \n",    nx);
      printf("   MPI tasks           =  %i \n",    numRanks);
      printf("   Iteration count     =  %i \n",    its);
      printf("   Final Origin Energy = %12.6e \n", e_zero);

      Real_t   MaxAbsDiff = Real_t(0.0);
      Real_t TotalAbsDiff = Real_t(0.0);
      Real_t   MaxRelDiff = Real_t(0.0);

      Real_t *e_all = new Real_t[nx * nx];
      cudaMemcpy(e_all, locDom.e.raw(), nx * nx * sizeof(Real_t), cudaMemcpyDeviceToHost);
      for (Index_t j=0; j<nx; ++j) {
         for (Index_t k=j+1; k<nx; ++k) {
            Real_t AbsDiff = FABS(e_all[j*nx+k]-e_all[k*nx+j]);
            TotalAbsDiff  += AbsDiff;

            if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

            Real_t RelDiff = AbsDiff / e_all[k*nx+j];

            if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
         }
      }
      delete e_all;

      // Quick symmetry check
      printf("   Testing Plane 0 of Energy Array on rank 0:\n");
      printf("        MaxAbsDiff   = %12.6e\n",   MaxAbsDiff   );
      printf("        TotalAbsDiff = %12.6e\n",   TotalAbsDiff );
      printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff   );
   }

   // Timing information
   printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
   printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n", grindTime1, grindTime2);
   printf("FOM                  = %10.8g (z/s)\n\n", 1000.0/grindTime2); // zones per second

   bool write_solution_flag=true;
   if (write_solution_flag) {
     write_solution(&locDom);
   }

   return ;
}

/* Main function - entry point for LULESH simulation 
 * Handles command-line arguments, initializes the domain, runs the simulation,
 * and outputs performance metrics
 */
int main(int argc, char *argv[])
{
  // Check for minimum required arguments (program name, grid type, size)
  if (argc < 3) {
    printUsage(argv);
    exit( LFileError );
  }
  
  // Validate the grid type argument: either -u (unstructured) or -s (structured)
  if ( strcmp(argv[1],"-u") != 0 && strcmp(argv[1],"-s") != 0 ) 
  {
    printUsage(argv);
    exit( LFileError ) ;
  }

  // Optional argument for max iterations (allows early termination)
  int num_iters = -1;
  if (argc == 5) {
    num_iters = atoi(argv[4]);
  }

  // Flag to indicate whether we're using structured or unstructured mesh
  bool structured = ( strcmp(argv[1],"-s") == 0 );

  // Variables for tracking MPI processes
  Int_t numRanks ;
  Int_t myRank ;

#if USE_MPI   
  Domain_member fieldData ;

  // Initialize MPI and get process info
  MPI_Init(&argc, &argv) ;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;  // Total number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;    // Current process ID
#else
  // For non-MPI builds, single process
  numRanks = 1;
  myRank = 0;
#endif

  // Initialize CUDA environment for this MPI rank
  cuda_init(myRank);

  /* assume cube subdomain geometry for now */
  // Number of elements in each dimension from command line
  Index_t nx = atoi(argv[2]);

  // Main domain object that holds all simulation data
  Domain *locDom ;

  // Set up the mesh and decompose. Assumes regular cubes for now
  // Calculate this process's position in the 3D grid of processes
  Int_t col, row, plane, side;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  // TODO: change default nr to 11
  // Domain region parameters for load balancing experiments
  Int_t nr = 11;      // Number of regions
  Int_t balance = 1;  // Region assignment algorithm
  Int_t cost = 1;     // Cost multiplier for evaluating equation of state

  // TODO: modify this constructor to account for new fields
  // TODO: setup communication buffers
  // Create the domain - sets up mesh, initializes physical variables
  locDom = NewDomain(argv, numRanks, col, row, plane, nx, side, structured, nr, balance, cost); 

#if USE_MPI   
   // copy to the host for mpi transfer
   // MPI communication requires data in host memory
   locDom->h_nodalMass = locDom->nodalMass;

   // Function pointer to access nodal mass data
   fieldData = &Domain::get_nodalMass;

   // Initial domain boundary communication 
   // Exchange ghost node data with neighboring processes
   CommRecv(*locDom, MSG_COMM_SBN, 1,
            locDom->sizeX + 1, locDom->sizeY + 1, locDom->sizeZ + 1,
            true, false) ;
   CommSend(*locDom, MSG_COMM_SBN, 1, &fieldData,
            locDom->sizeX + 1, locDom->sizeY + 1, locDom->sizeZ + 1,
            true, false) ;
   CommSBN(*locDom, 1, &fieldData) ;

   // copy back to the device
   // After MPI exchange, move updated data back to GPU
   locDom->nodalMass = locDom->h_nodalMass;

   // End initialization
   // Wait for all processes to complete initialization
   MPI_Barrier(MPI_COMM_WORLD);
#endif

  // Set CUDA cache preference to favor L1 cache for better performance
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* timestep to solution */
  // Iteration counter
  int its=0;

  // Print simulation parameters (only from rank 0)
  if (myRank == 0) {
    if (structured)
      printf("Running until t=%f, Problem size=%dx%dx%d\n",locDom->stoptime,nx,nx,nx);
    else 
      printf("Running until t=%f, Problem size=%d \n",locDom->stoptime,locDom->numElem);
  }

  // Start CUDA profiling for performance analysis
  cudaProfilerStart();

  // Start timer for measuring performance
#if USE_MPI   
   double start = MPI_Wtime();
#else
   timeval start;
   gettimeofday(&start, NULL) ;
#endif

  // Main simulation loop - continues until simulation time reaches stop time
  while(locDom->time_h < locDom->stoptime)
  {
    // this has been moved after computation of volume forces to hide launch latencies
    //TimeIncrement(locDom) ;

    // Execute one timestep of the Lagrangian hydrodynamics simulation
    LagrangeLeapFrog(locDom) ;

    // Verify solution is still valid, handles errors if any
    checkErrors(locDom,its,myRank);

    // Optionally print progress information
    #if LULESH_SHOW_PROGRESS
     if (myRank == 0) 
	 printf("cycle = %d, time = %e, dt=%e\n", its+1, double(locDom->time_h), double(locDom->deltatime_h) ) ;
    #endif
    its++;
    // Exit early if we've reached the specified iteration limit
    if (its == num_iters) break;
  }

  // make sure GPU finished its work
  // Synchronize to ensure all GPU operations are complete
  cudaDeviceSynchronize();

// Use reduced max elapsed time
   // Calculate elapsed time for this process
   double elapsed_time;
#if USE_MPI   
   elapsed_time = MPI_Wtime() - start;
#else
   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
#endif

   // For MPI, find the maximum time across all processes
   // (overall performance is limited by slowest process)
   double elapsed_timeG;
#if USE_MPI   
   MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE,
              MPI_MAX, 0, MPI_COMM_WORLD);
#else
   elapsed_timeG = elapsed_time;
#endif

  // Stop CUDA profiling
  cudaProfilerStop();

  // Verify results and output performance metrics (only from rank 0)
  if (myRank == 0) 
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, its, nx, numRanks, structured);

#ifdef SAMI
  // Optional: dump domain data for visualization
  DumpDomain(locDom) ;
#endif
  // Reset CUDA device to clean state
  cudaDeviceReset();

#if USE_MPI
   // Finalize MPI
   MPI_Finalize() ;
#endif

  return 0 ;
}

