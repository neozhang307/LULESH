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
#include "lulesh_computation.h"
#include "utility/util.h"
#include "utility/sm_utils.inl"
// #include "utility/allocator.h"
// Do not include lulesh_kernels.h here to avoid duplicate definitions
// checkErrors function is already defined at line 75






// static inline
/*
 * LagrangeNodal - Performs node-centered calculations for one timestep
 *
 * This function handles the first phase of the Lagrangian simulation:
 * 1. Calculate forces at nodes from element contributions
 * 2. Compute nodal accelerations based on forces and masses
 * 3. Apply boundary conditions (e.g., symmetry, free surfaces)
 * 4. Update node positions and velocities
 * 5. Synchronize updated values between processes (for MPI)
 *
 * This phase focuses on the mesh nodes (vertices) rather than elements.
 */
void LagrangeNodal(Domain *domain)
{
  printf("DEBUG: Inside LagrangeNodal\n");
  
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   // Array of function pointers for accessing domain data during MPI communication
   Domain_member fieldData[6];
   printf("DEBUG: SEDOV_SYNC_POS_VEL_EARLY defined\n");
#endif

  // Get bulk viscosity cutoff parameter (controls artificial viscosity)
  Real_t u_cut = domain->u_cut;
  printf("DEBUG: Got u_cut = %e\n", u_cut);

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  // Step 1: Calculate forces on each node from surrounding elements
  // This accumulates all forces (internal, external, artificial viscosity)
  printf("DEBUG: Calling CalcForceForNodes\n");
  CalcForceForNodes(domain);
  printf("DEBUG: CalcForceForNodes completed\n");

#if USE_MPI  
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   // For MPI: Start receiving position and velocity data from other processes
   // This is done early to overlap communication with computation
   CommRecv(*domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
            false, false);
#endif
#endif

  // Step 2: Calculate accelerations for all nodes (F = ma -> a = F/m)
  CalcAccelerationForNodes(domain);

  // Step 3: Apply boundary conditions to accelerations
  // This enforces constraints like symmetry planes and free surfaces
  ApplyAccelerationBoundaryConditionsForNodes(domain);

  // Step 4: Update node positions and velocities using calculated accelerations
  // This applies the time integration scheme (leapfrog method)
  CalcPositionAndVelocityForNodes(u_cut, domain);

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  // For MPI: Synchronize position and velocity data with neighboring processes
  
  // initialize raw device pointers for MPI communication
  domain->d_x = domain->x;
  domain->d_y = domain->y;
  domain->d_z = domain->z;

  domain->d_xd = domain->xd;
  domain->d_yd = domain->yd;
  domain->d_zd = domain->zd;

  // Set up function pointers to access position and velocity components
  fieldData[0] = &Domain::get_x;  // x-position
  fieldData[1] = &Domain::get_y;  // y-position
  fieldData[2] = &Domain::get_z;  // z-position
  fieldData[3] = &Domain::get_xd; // x-velocity
  fieldData[4] = &Domain::get_yd; // y-velocity
  fieldData[5] = &Domain::get_zd; // z-velocity

  // Send position and velocity data directly from GPU to other processes
  CommSendGpu(*domain, MSG_SYNC_POS_VEL, 6, fieldData,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           false, false, domain->streams[2]);
  
  // Complete synchronization of position and velocity data
  CommSyncPosVelGpu(*domain, &domain->streams[2]);
#endif
#endif

  return;
}
// checkErrors is already defined above

void CalcForceForNodes(Domain *domain)
{
#if USE_MPI  
  CommRecv(*domain, MSG_COMM_SBN, 3,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           true, false) ;
#endif

  CalcVolumeForceForElems(domain);

  // moved here from the main loop to allow async execution with GPU work
  TimeIncrement(domain);

#if USE_MPI 
  // initialize pointers
  domain->d_fx = domain->fx;
  domain->d_fy = domain->fy;
  domain->d_fz = domain->fz;

  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::get_fx ;
  fieldData[1] = &Domain::get_fy ;
  fieldData[2] = &Domain::get_fz ;

  CommSendGpu(*domain, MSG_COMM_SBN, 3, fieldData,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           true, false, domain->streams[2]) ;
  CommSBNGpu(*domain, 3, fieldData, &domain->streams[2]) ;
#endif
}


// static inline
/*
 * LagrangeElements - Performs element-centered calculations for one timestep
 *
 * This function handles the second phase of the Lagrangian simulation, focusing on
 * element (cell) calculations:
 * 1. Calculate kinematics (velocity gradients, strain rates)
 * 2. Compute artificial viscosity (q) for shock handling
 * 3. Apply material properties (equation of state)
 * 4. Update element volumes and related quantities
 *
 * The function manages temporary memory allocation/deallocation and MPI communication
 * of element-centered quantities between processes.
 */
void LagrangeElements(Domain *domain)
{
  printf("DEBUG: Inside LagrangeElements\n");
  fflush(stdout);
  
  // Calculate total elements including ghost elements for MPI
  int allElem = domain->numElem +  /* local elem */
                2*domain->sizeX*domain->sizeY + /* plane ghosts */
                2*domain->sizeX*domain->sizeZ + /* row ghosts */
                2*domain->sizeY*domain->sizeZ ; /* col ghosts */
  
  printf("DEBUG: Domain dimensions - sizeX=%d, sizeY=%d, sizeZ=%d\n", 
         domain->sizeX, domain->sizeY, domain->sizeZ);
  printf("DEBUG: Elements - local=%d, total with ghosts=%d\n", 
         domain->numElem, allElem);
  fflush(stdout);

  // Allocate temporary arrays for this phase
  printf("DEBUG: Allocating temporary arrays\n");
  fflush(stdout);
  
  // vnew: new relative volume
  cudaMalloc((void**)&domain->vnew, domain->numElem * sizeof(Real_t));
  // Principal strain terms (diagonal components of strain tensor)
  cudaMalloc((void**)&domain->dxx, domain->numElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->dyy, domain->numElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->dzz, domain->numElem * sizeof(Real_t));

  // Coordinate gradients (spatial derivatives in each direction)
  cudaMalloc((void**)&domain->delx_xi, domain->numElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->delx_eta, domain->numElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->delx_zeta, domain->numElem * sizeof(Real_t));

  // Velocity gradients (for all elements including ghosts)
  cudaMalloc((void**)&domain->delv_xi, allElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->delv_eta, allElem * sizeof(Real_t));
  cudaMalloc((void**)&domain->delv_zeta, allElem * sizeof(Real_t));
  
  printf("DEBUG: Temporary arrays allocated successfully\n");
  fflush(stdout);

#if USE_MPI     
  // For MPI: Start receiving monotonic q gradient data from other processes
  CommRecv(*domain, MSG_MONOQ, 3,
           domain->sizeX, domain->sizeY, domain->sizeZ,
           true, true);
  printf("DEBUG: MPI - Started receiving monotonic q gradient data\n");
  fflush(stdout);
#endif

  /*********************************************/
  /*  Calc Kinematics and Monotic Q Gradient   */
  /*********************************************/
  // Step A: Calculate velocity gradients and related terms
  // This computes how quickly the element is deforming
  printf("DEBUG: Calling CalcKinematicsAndMonotonicQGradient\n");
  fflush(stdout);
  
  // Check if volo pointer is valid before calculation
  if (domain->volo == nullptr) {
    printf("ERROR: domain->volo is NULL before CalcKinematicsAndMonotonicQGradient\n");
    fflush(stdout);
  } else {
    // Read first element of volo for debugging
    Real_t firstVolo;
    cudaError_t err = cudaMemcpy(&firstVolo, domain->volo, sizeof(Real_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      printf("ERROR: Failed to read volo[0]: %s\n", cudaGetErrorString(err));
    } else {
      printf("DEBUG: First element volo[0] = %e before kinematics calculation\n", firstVolo);
    }
    fflush(stdout);
  }
  
  CalcKinematicsAndMonotonicQGradient(domain);
  
  printf("DEBUG: CalcKinematicsAndMonotonicQGradient completed\n");
  fflush(stdout);

#if USE_MPI      
   // For MPI: Send calculated gradient data to other processes
   Domain_member fieldData[3];

   printf("DEBUG: MPI - Preparing to send monotonic q gradient data\n");
   fflush(stdout);
   
   // Initialize raw device pointers for MPI communication
   domain->d_delv_xi = domain->delv_xi;
   domain->d_delv_eta = domain->delv_eta;
   domain->d_delv_zeta = domain->delv_zeta;

   // Set up function pointers for velocity gradient access
   fieldData[0] = &Domain::get_delv_xi;
   fieldData[1] = &Domain::get_delv_eta;
   fieldData[2] = &Domain::get_delv_zeta;

   // Send velocity gradients directly from GPU to other processes
   CommSendGpu(*domain, MSG_MONOQ, 3, fieldData,
            domain->sizeX, domain->sizeY, domain->sizeZ,
            true, true, domain->streams[2]);
   
   // Complete monotonic q communication
   CommMonoQGpu(*domain, domain->streams[2]);
   
   printf("DEBUG: MPI - Completed monotonic q communication\n");
   fflush(stdout);
#endif

  // Free temporary arrays no longer needed
  printf("DEBUG: Freeing first batch of temporary arrays (dxx, dyy, dzz)\n");
  fflush(stdout);
  
  cudaFree(domain->dxx);
  cudaFree(domain->dyy);
  cudaFree(domain->dzz);

  /**********************************
  *    Calc Monotic Q Region
  **********************************/
  // Step B: Calculate artificial viscosity (q) for shock treatment
  // This helps prevent numerical oscillations near shock fronts
  printf("DEBUG: Calling CalcMonotonicQRegionForElems\n");
  fflush(stdout);
  
  // Check domain->volo again
  if (domain->volo != nullptr) {
    Real_t firstVolo;
    cudaError_t err = cudaMemcpy(&firstVolo, domain->volo, sizeof(Real_t), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
      printf("DEBUG: First element volo[0] = %e before MonotonicQ calculation\n", firstVolo);
      fflush(stdout);
    }
  }
  
  CalcMonotonicQRegionForElems(domain);
  
  printf("DEBUG: CalcMonotonicQRegionForElems completed\n");
  fflush(stdout);

  // Free more temporary arrays
  printf("DEBUG: Freeing second batch of temporary arrays (delx_*, delv_*)\n");
  fflush(stdout);
  
  cudaFree(domain->delx_xi);
  cudaFree(domain->delx_eta);
  cudaFree(domain->delx_zeta);

  cudaFree(domain->delv_xi);
  cudaFree(domain->delv_eta);
  cudaFree(domain->delv_zeta);

  // Step C: Apply material equation of state and update element volumes
  // This calculates new pressures, energies, and volumes based on the
  // material model (ideal gas for Sedov blast wave problem)
  printf("DEBUG: Calling ApplyMaterialPropertiesAndUpdateVolume\n");
  fflush(stdout);
  
  // Check domain->volo once more
  if (domain->volo != nullptr) {
    Real_t firstVolo;
    cudaError_t err = cudaMemcpy(&firstVolo, domain->volo, sizeof(Real_t), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess) {
      printf("DEBUG: First element volo[0] = %e before material properties application\n", firstVolo);
      fflush(stdout);
    }
  }
  
  ApplyMaterialPropertiesAndUpdateVolume(domain);
  
  printf("DEBUG: ApplyMaterialPropertiesAndUpdateVolume completed\n");
  fflush(stdout);
  
  // Free the last temporary array
  printf("DEBUG: Freeing last temporary array (vnew)\n");
  cudaFree(domain->vnew);
  printf("DEBUG: LagrangeElements function completed\n");
  fflush(stdout);
}
// LagrangeNodal function already defined at line 1824

// LagrangeElements function already defined at line 2950

// Time increment calculation
void TimeIncrement(Domain* domain)
{
    // To make sure dtcourant and dthydro have been updated on host
    cudaEventSynchronize(domain->time_constraint_computed);

    Real_t targetdt = domain->stoptime - domain->time_h;

    if ((domain->dtfixed <= Real_t(0.0)) && (domain->cycle != Int_t(0))) {

      Real_t ratio;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20);
      Real_t newdt;
      if (*(domain->dtcourant_h) < gnewdt) { 
         gnewdt = *(domain->dtcourant_h) / Real_t(2.0);
      }
      if (*(domain->dthydro_h) < gnewdt) { 
         gnewdt = *(domain->dthydro_h) * Real_t(2.0) / Real_t(3.0);
      }

#if USE_MPI      
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD);
#else
      newdt = gnewdt;
#endif

      Real_t olddt = domain->deltatime_h;
      ratio = newdt / olddt;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain->deltatimemultlb) {
            newdt = olddt;
         }
         else if (ratio > domain->deltatimemultub) {
            newdt = olddt*domain->deltatimemultub;
         }
      }

      if (newdt > domain->dtmax) {
         newdt = domain->dtmax;
      }
      domain->deltatime_h = newdt;
    }

    /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
    if ((targetdt > domain->deltatime_h) &&
         (targetdt < (Real_t(4.0) * domain->deltatime_h / Real_t(3.0)))) {
       targetdt = Real_t(2.0) * domain->deltatime_h / Real_t(3.0);
    }

    if (targetdt < domain->deltatime_h) {
       domain->deltatime_h = targetdt;
    }

    domain->time_h += domain->deltatime_h;

    domain->cycle++;
}



// Lagrangian leap-frog time integration
void LagrangeLeapFrog(Domain* domain)
{
   printf("DEBUG: Inside LagrangeLeapFrog\n");
   
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   // Phase 1: Update node-centered quantities
   // - Calculates forces at nodes from element contributions
   // - Computes accelerations based on forces and nodal masses
   // - Applies boundary conditions (symmetry, free surfaces)
   // - Updates positions and velocities using time integration
   printf("DEBUG: Calling LagrangeNodal\n");
   LagrangeNodal(domain);
   printf("DEBUG: LagrangeNodal completed\n");

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   // Phase 2: Update element-centered quantities
   // - Computes velocity gradients for elements
   // - Calculates artificial viscosity (q) for shock treatment
   // - Updates element volumes and material states
   // - Applies equation of state to compute new pressures and energies
   printf("DEBUG: Calling LagrangeElements\n");
   LagrangeElements(domain);
   printf("DEBUG: LagrangeElements completed\n");

   // Phase 3: Calculate new timestep based on Courant-Friedrichs-Lewy (CFL) condition
   // - Computes maximum stable timestep to ensure simulation stability
   // - Uses both velocity (Courant) and volume change (hydro) constraints
   // - Selects the most restrictive timestep across all elements
   printf("DEBUG: Calling CalcTimeConstraintsForElems\n");
   CalcTimeConstraintsForElems(domain);
   printf("DEBUG: CalcTimeConstraintsForElems completed\n");
}
