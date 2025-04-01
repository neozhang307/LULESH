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
#include "lulesh_comm.h"
#include "utility/util.h"
#include "utility/sm_utils.inl"
#include "utility/allocator.h"
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
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   // Array of function pointers for accessing domain data during MPI communication
   Domain_member fieldData[6];
#endif

  // Get bulk viscosity cutoff parameter (controls artificial viscosity)
  Real_t u_cut = domain->u_cut;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  // Step 1: Calculate forces on each node from surrounding elements
  // This accumulates all forces (internal, external, artificial viscosity)
  CalcForceForNodes(domain);

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
  domain->d_x = domain->x.raw();
  domain->d_y = domain->y.raw();
  domain->d_z = domain->z.raw();

  domain->d_xd = domain->xd.raw();
  domain->d_yd = domain->yd.raw();
  domain->d_zd = domain->zd.raw();

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
  domain->d_fx = domain->fx.raw();
  domain->d_fy = domain->fy.raw();
  domain->d_fz = domain->fz.raw();

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
  // Calculate total elements including ghost elements for MPI
  int allElem = domain->numElem +  /* local elem */
                2*domain->sizeX*domain->sizeY + /* plane ghosts */
                2*domain->sizeX*domain->sizeZ + /* row ghosts */
                2*domain->sizeY*domain->sizeZ ; /* col ghosts */

  // Allocate temporary arrays for this phase
  // vnew: new relative volume
  domain->vnew = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  // Principal strain terms (diagonal components of strain tensor)
  domain->dxx  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->dyy  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->dzz  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);

  // Coordinate gradients (spatial derivatives in each direction)
  domain->delx_xi    = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->delx_eta   = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->delx_zeta  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);

  // Velocity gradients (for all elements including ghosts)
  domain->delv_xi    = Allocator< Vector_d<Real_t> >::allocate(allElem);
  domain->delv_eta   = Allocator< Vector_d<Real_t> >::allocate(allElem);
  domain->delv_zeta  = Allocator< Vector_d<Real_t> >::allocate(allElem);

#if USE_MPI     
  // For MPI: Start receiving monotonic q gradient data from other processes
  CommRecv(*domain, MSG_MONOQ, 3,
           domain->sizeX, domain->sizeY, domain->sizeZ,
           true, true);
#endif

  /*********************************************/
  /*  Calc Kinematics and Monotic Q Gradient   */
  /*********************************************/
  // Step A: Calculate velocity gradients and related terms
  // This computes how quickly the element is deforming
  CalcKinematicsAndMonotonicQGradient(domain);

#if USE_MPI      
   // For MPI: Send calculated gradient data to other processes
   Domain_member fieldData[3];

   // Initialize raw device pointers for MPI communication
   domain->d_delv_xi = domain->delv_xi->raw();
   domain->d_delv_eta = domain->delv_eta->raw();
   domain->d_delv_zeta = domain->delv_zeta->raw();

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
#endif

  // Free temporary arrays no longer needed
  Allocator<Vector_d<Real_t> >::free(domain->dxx, domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->dyy, domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->dzz, domain->numElem);

  /**********************************
  *    Calc Monotic Q Region
  **********************************/
  // Step B: Calculate artificial viscosity (q) for shock treatment
  // This helps prevent numerical oscillations near shock fronts
   CalcMonotonicQRegionForElems(domain);

  // Free more temporary arrays
  Allocator<Vector_d<Real_t> >::free(domain->delx_xi, domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->delx_eta, domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->delx_zeta, domain->numElem);

  Allocator<Vector_d<Real_t> >::free(domain->delv_xi, allElem);
  Allocator<Vector_d<Real_t> >::free(domain->delv_eta, allElem);
  Allocator<Vector_d<Real_t> >::free(domain->delv_zeta, allElem);

  // Step C: Apply material equation of state and update element volumes
  // This calculates new pressures, energies, and volumes based on the
  // material model (ideal gas for Sedov blast wave problem)
  ApplyMaterialPropertiesAndUpdateVolume(domain);
  
  // Free the last temporary array
  Allocator<Vector_d<Real_t> >::free(domain->vnew, domain->numElem);
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
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   // Phase 1: Update node-centered quantities
   // - Calculates forces at nodes from element contributions
   // - Computes accelerations based on forces and nodal masses
   // - Applies boundary conditions (symmetry, free surfaces)
   // - Updates positions and velocities using time integration
   LagrangeNodal(domain);

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   // Phase 2: Update element-centered quantities
   // - Computes velocity gradients for elements
   // - Calculates artificial viscosity (q) for shock treatment
   // - Updates element volumes and material states
   // - Applies equation of state to compute new pressures and energies
   LagrangeElements(domain);

   // Phase 3: Calculate new timestep based on Courant-Friedrichs-Lewy (CFL) condition
   // - Computes maximum stable timestep to ensure simulation stability
   // - Uses both velocity (Courant) and volume change (hydro) constraints
   // - Selects the most restrictive timestep across all elements
   CalcTimeConstraintsForElems(domain);
}
