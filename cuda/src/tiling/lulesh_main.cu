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
#include <cuda.h>

#include <utility/util.h>
#include <utility/sm_utils.inl>
#include <utility/allocator.h>
#include "cuda_profiler_api.h"

#include "lulesh_split.h"
#ifdef USE_TILING
#include "lulesh_domain_groups.h"
#endif



// Print usage information for command line
void printUsage(char *argv[])
{
  printf("Usage: \n");
  printf("Unstructured grid:  %s -u <file.lmesh> [-t tileSize]\n", argv[0]);
  printf("Structured grid:    %s -s numEdgeElems [-t tileSize]\n", argv[0]);
  printf("\nOptions:\n");
  printf("  -t tileSize : Set tiling dimensions (defaults to 1,1,1 if not specified)\n");
  printf("\nExamples:\n");
  printf("%s -s 45\n", argv[0]);
  printf("%s -s 45 -t 2\n", argv[0]);
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

  // Default tiling size (1 means no tiling)
  Index_t tileSize = 1;
  
  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
      tileSize = atoi(argv[i+1]);
      i++; // Skip the next argument as we've already processed it
    }
  }
  
  // Optional argument for max iterations (allows early termination)
  int num_iters = -1;
  // Look for iterations parameter after processing other arguments
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i+1 < argc) {
      num_iters = atoi(argv[i+1]);
      i++; // Skip the next argument
    }
  }

  // Flag to indicate whether we're using structured or unstructured mesh
  bool structured = ( strcmp(argv[1],"-s") == 0 );

  // Variables for tracking MPI processes
  Int_t numRanks ;
  Int_t myRank ;


  // For non-MPI builds, single process
  numRanks = 1;
  myRank = 0;


  // Initialize CUDA environment for this MPI rank
  cuda_init(myRank);

  /* assume cube subdomain geometry for now */
  // Number of elements in each dimension from command line
  Index_t nx = atoi(argv[2]);

  // Domain group for managing domains
  DomainGroup *domainGroup = nullptr;
  
  // Base domain reference for compatibility with existing code
  Domain *baseDom = nullptr;

  // Print tiling information if applicable
  if (myRank == 0 && tileSize > 1) {
    printf("Running with tiling size %d x %d x %d\n", tileSize, tileSize, tileSize);
  }
  
  // TODO: change default nr to 11
  // Domain region parameters for load balancing experiments
  Int_t nr = 11;      // Number of regions
  Int_t balance = 1;  // Region assignment algorithm
  Int_t cost = 1;     // Cost multiplier for evaluating equation of state

  // Create a domain group to manage the domains
  domainGroup = new DomainGroup(tileSize, tileSize, tileSize);
  
  // Initialize all domains in the group
  domainGroup->initializeDomains(argv, nx, structured, nr, balance, cost);
  
  // Get the first domain (0,0,0) as base domain for compatibility with existing code
  baseDom = domainGroup->getDomain(0, 0, 0);

#if USE_MPI   
   // copy to the host for mpi transfer
   // MPI communication requires data in host memory
   baseDom->h_nodalMass = baseDom->nodalMass;

   // Function pointer to access nodal mass data
   fieldData = &Domain::get_nodalMass;

   // Initial domain boundary communication 
   // Exchange ghost node data with neighboring processes
   CommRecv(*baseDom, MSG_COMM_SBN, 1,
            baseDom->sizeX + 1, baseDom->sizeY + 1, baseDom->sizeZ + 1,
            true, false) ;
   CommSend(*baseDom, MSG_COMM_SBN, 1, &fieldData,
            baseDom->sizeX + 1, baseDom->sizeY + 1, baseDom->sizeZ + 1,
            true, false) ;
   CommSBN(*baseDom, 1, &fieldData) ;

   // copy back to the device
   // After MPI exchange, move updated data back to GPU
   baseDom->nodalMass = baseDom->h_nodalMass;

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
  // if (myRank == 0) 
  {
    if (structured)
      printf("Running until t=%f, Problem size=%dx%dx%d\n",baseDom->stoptime,nx,nx,nx);
    else 
      printf("Running until t=%f, Problem size=%d \n",baseDom->stoptime,baseDom->numElem);
  }

  // Start CUDA profiling for performance analysis
  cudaProfilerStart();

  // Start timer for measuring performance

   timeval start;
   gettimeofday(&start, NULL) ;

  // Main simulation loop - continues until simulation time reaches stop time
  while(baseDom->time_h < baseDom->stoptime)
  {
    // For tiled execution, we could use domainGroup->stepSimulation() here
    // But for now, we'll continue with the existing approach using baseDom
    
    // Execute one timestep of the Lagrangian hydrodynamics simulation
    LagrangeLeapFrog(baseDom);

    // Verify solution is still valid, handles errors if any
    checkErrors(baseDom, its, myRank);

    // Optionally print progress information
    #if LULESH_SHOW_PROGRESS
     if (myRank == 0) 
	 printf("cycle = %d, time = %e, dt=%e\n", its+1, double(baseDom->time_h), double(baseDom->deltatime_h));
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

   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;

   // For MPI, find the maximum time across all processes
   // (overall performance is limited by slowest process)
   double elapsed_timeG;

   elapsed_timeG = elapsed_time;

  // Stop CUDA profiling
  cudaProfilerStop();

  // Verify results and output performance metrics (only from rank 0)
  if (myRank == 0) 
    VerifyAndWriteFinalOutput(elapsed_timeG, *baseDom, its, nx, numRanks, structured);

#ifdef SAMI
  // Optional: dump domain data for visualization
  DumpDomain(baseDom) ;
#endif

  // Clean up the domain group
  delete domainGroup;
  // Reset CUDA device to clean state
  cudaDeviceReset();


  return 0 ;
}

