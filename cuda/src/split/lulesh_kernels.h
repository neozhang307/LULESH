#pragma once
/****************************************************/
/* Allow flexibility for arithmetic representations */
/****************************************************/
__device__ inline real4 SQRT(real4 arg) { return sqrtf(arg); }
__device__ inline real8 SQRT(real8 arg) { return sqrt(arg); }

__device__ inline real4 CBRT(real4 arg) { return cbrtf(arg); }
__device__ inline real8 CBRT(real8 arg) { return cbrt(arg); }

// FABS is now defined in lulesh.h

__device__ inline real4 FMAX(real4 arg1, real4 arg2) { return fmaxf(arg1, arg2); }
__device__ inline real8 FMAX(real8 arg1, real8 arg2) { return fmax(arg1, arg2); }

#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)		\
{									\
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;			\
									\
   dvdc= 								\
     ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) +	\
     ((a0) + (a4)) * ((b3) + (b4)) - ((a3) + (a4)) * ((b0) + (b4)) -	\
     ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));	\
   dvdc *= twelfth;							\
}
/******************************************/
/* Utility CUDA device helper functions */
/******************************************/
__device__
static 
__forceinline__
void SumOverNodesShfl(Real_t& val) {
   val += utils::shfl_xor(val, 4, 8);
   val += utils::shfl_xor(val, 2, 8);
   val += utils::shfl_xor(val, 1, 8);
}

/******************************************/
/* Volume calculation kernels */
/******************************************/
__host__ __device__
static 
__forceinline__ 
Real_t CalcElemVolume(const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7)
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

// CalcElemVolume wrapper that takes arrays for coordinates
__host__ __device__
// inline
Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8])
{
  return CalcElemVolume(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                        z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}




__device__
static 
 __forceinline__
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  /*

  b[0][4] = - cjxxi  -  cjxet  +  cjxze;
  b[0][5] = + cjxxi  -  cjxet  +  cjxze;
  b[0][6] = + cjxxi  +  cjxet  +  cjxze;
  b[0][7] = - cjxxi  +  cjxet  +  cjxze;

  */

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static
__device__
__forceinline__
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

static
__device__
__forceinline__
void SumElemFaceNormal_warp_per_4cell(
                       Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       const Real_t x, const Real_t y, const Real_t z,
                       int node, 
                       int n0, int n1, int n2, int n3)
{
  Real_t coef0 = Real_t(0.5);
  Real_t coef1 = Real_t(0.5);
  if (node == n0 || node == n1 || node==n2 || node==n3)
  {
    if (node == n0 || node == n1)
       coef0 = -coef0;

    if (node == n0 || node == n3)
       coef1 = -coef1;
  }
  else
  {
    coef0 = Real_t(0.);
    coef1 = Real_t(0.);
  }
   
   Real_t bisectX0 = coef0*x;
   Real_t bisectY0 = coef0*y;
   Real_t bisectZ0 = coef0*z;

   Real_t bisectX1 = coef1*x;
   Real_t bisectY1 = coef1*y;
   Real_t bisectZ1 = coef1*z;

   SumOverNodesShfl(bisectX0);
   SumOverNodesShfl(bisectY0);
   SumOverNodesShfl(bisectZ0);

   SumOverNodesShfl(bisectX1);
   SumOverNodesShfl(bisectY1);
   SumOverNodesShfl(bisectZ1);

   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   if (node == n0 || node == n1 || node==n2 || node==n3)
   {
     *normalX0 += areaX;
     *normalY0 += areaY;
     *normalZ0 += areaZ;
   }

}

__device__
static inline
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}



/*
 * AddNodeForcesFromElems_kernel - CUDA kernel to distribute element forces to nodes
 *
 * This kernel performs a key step in Lagrangian hydrodynamics: calculating the
 * nodal forces by accumulating contributions from all elements connected to each node.
 * 
 * The kernel processing works as follows:
 * 1. Each thread handles one node
 * 2. For each node, it identifies all connected elements using connectivity arrays
 * 3. It accumulates force contributions from each connected element
 * 4. Finally, it stores the total force for the node
 */
__global__
void AddNodeForcesFromElems_kernel( Index_t numNode,          // Total number of nodes
                                    Index_t padded_numNode,    // Padded node count for memory alignment
                                    const Int_t* nodeElemCount,  // Number of elements connected to each node
                                    const Int_t* nodeElemStart,  // Starting index in nodeElemCornerList for each node
                                    const Index_t* nodeElemCornerList, // Map from node to element corners
                                    const Real_t* fx_elem,     // X-component of element forces
                                    const Real_t* fy_elem,     // Y-component of element forces
                                    const Real_t* fz_elem,     // Z-component of element forces
                                    Real_t* fx_node,           // Output: X-component of nodal forces
                                    Real_t* fy_node,           // Output: Y-component of nodal forces
                                    Real_t* fz_node,           // Output: Z-component of nodal forces
                                    const Int_t num_threads)   // Total number of threads to launch
{
    // Calculate global thread ID
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    
    // Only process valid nodes
    if (tid < num_threads)
    {
      // Get global node index
      Index_t g_i = tid;
      
      // Get number of elements connected to this node
      Int_t count = nodeElemCount[g_i];
      
      // Get starting index in the corner list for this node
      Int_t start = nodeElemStart[g_i];
      
      // Initialize force accumulators
      Real_t fx, fy, fz;
      fx = fy = fz = Real_t(0.0);

      // Loop through all elements connected to this node
      for (int j=0; j<count; j++) 
      {
          // Find the element corner position
          // Note: This access pattern can be uncoalesced, potentially hurting performance
          Index_t pos = nodeElemCornerList[start+j]; // Uncoalesced access here
          
          // Accumulate force contributions from this element
          fx += fx_elem[pos]; 
          fy += fy_elem[pos]; 
          fz += fz_elem[pos];
      }

      // Store the calculated forces back to global memory
      fx_node[g_i] = fx; 
      fy_node[g_i] = fy; 
      fz_node[g_i] = fz;
    }
}

static
__device__
__forceinline__
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);

   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

static
__device__
__forceinline__
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

static
__device__ 
__forceinline__
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
                              Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
                              Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
                              Real_t *hourgam7, Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}

__device__
__forceinline__
void CalcHourglassModes(const Real_t xn[8], const Real_t yn[8], const Real_t zn[8],
                        const Real_t dvdxn[8], const Real_t dvdyn[8], const Real_t dvdzn[8],
                        Real_t hourgam[8][4], Real_t volinv)
{
    Real_t hourmodx, hourmody, hourmodz;

    hourmodx = xn[0] + xn[1] - xn[2] - xn[3] - xn[4] - xn[5] + xn[6] + xn[7];
    hourmody = yn[0] + yn[1] - yn[2] - yn[3] - yn[4] - yn[5] + yn[6] + yn[7];
    hourmodz = zn[0] + zn[1] - zn[2] - zn[3] - zn[4] - zn[5] + zn[6] + zn[7]; // 21
    hourgam[0][0] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][0] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][0] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][0] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][0] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][0] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][0] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][0] =  1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz); // 60

    hourmodx = xn[0] - xn[1] - xn[2] + xn[3] - xn[4] + xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] - yn[2] + yn[3] - yn[4] + yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] - zn[2] + zn[3] - zn[4] + zn[5] + zn[6] - zn[7];
    hourgam[0][1] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][1] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][1] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][1] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][1] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][1] =  1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][1] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][1] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = xn[0] - xn[1] + xn[2] - xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] + yn[2] - yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] + zn[2] - zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][2] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][2] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][2] =  1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][2] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][2] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][2] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][2] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][2] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = -xn[0] + xn[1] - xn[2] + xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = -yn[0] + yn[1] - yn[2] + yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = -zn[0] + zn[1] - zn[2] + zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][3] = -1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][3] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][3] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][3] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][3] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][3] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][3] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][3] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

}


template< bool hourg_gt_zero > 
__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(64,4) 
#else
__launch_bounds__(64,8) 
#endif
void CalcVolumeForceForElems_kernel(

    const Real_t* __restrict__ volo, 
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p, 
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss, 
    const Real_t* __restrict__ elemMass,
    const Real_t* __restrict__ x,   const Real_t* __restrict__ y,  const Real_t* __restrict__  z,
    const Real_t* __restrict__ xd,  const Real_t* __restrict__ yd,  const Real_t* __restrict__  zd,
    //TextureObj<Real_t> x,  TextureObj<Real_t> y,  TextureObj<Real_t> z,
    //TextureObj<Real_t> xd,  TextureObj<Real_t> yd,  TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x,  TextureObj<Real_t>* y,  TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd,  TextureObj<Real_t>* yd,  TextureObj<Real_t>* zd,
#ifdef DOUBLE_PRECISION // For floats, use atomicAdd
    Real_t* __restrict__ fx_elem, 
    Real_t* __restrict__ fy_elem, 
    Real_t* __restrict__ fz_elem,
#else
    Real_t* __restrict__ fx_node, 
    Real_t* __restrict__ fy_node, 
    Real_t* __restrict__ fz_node,
#endif
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads)

{

  /*************************************************
  *     FUNCTION: Calculates the volume forces
  *************************************************/

  Real_t xn[8],yn[8],zn[8];;
  Real_t xdn[8],ydn[8],zdn[8];;
  Real_t dvdxn[8],dvdyn[8],dvdzn[8];;
  Real_t hgfx[8],hgfy[8],hgfz[8];;
  Real_t hourgam[8][4];
  Real_t coefficient;

  int elem=blockDim.x*blockIdx.x+threadIdx.x;
  if (elem < num_threads) 
  {
    Real_t volume = v[elem];
    Real_t det = volo[elem] * volume;

    // Check for bad volume
    if (volume < 0.) {
      *bad_vol = elem; 
    }

    Real_t ss1 = ss[elem];
    Real_t mass1 = elemMass[elem];
    Real_t sigxx = -p[elem] - q[elem];

    Index_t n[8];
    #pragma unroll 
    for (int i=0;i<8;i++) {
      n[i] = nodelist[elem+i*padded_numElem];
    }

    Real_t volinv = Real_t(1.0) / det;
    //#pragma unroll 
    //for (int i=0;i<8;i++) {
    //  xn[i] =x[n[i]];
    //  yn[i] =y[n[i]];
    //  zn[i] =z[n[i]];
    //}

    #pragma unroll 
    for (int i=0;i<8;i++)
      xn[i] =x[n[i]];

    #pragma unroll 
    for (int i=0;i<8;i++)
      yn[i] =y[n[i]];

    #pragma unroll 
    for (int i=0;i<8;i++)
      zn[i] =z[n[i]];


    Real_t volume13 = CBRT(det);
    coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    /*************************************************/
    /*    compute the volume derivatives             */
    /*************************************************/
    CalcElemVolumeDerivative(dvdxn, dvdyn, dvdzn, xn, yn, zn); 

    /*************************************************/
    /*    compute the hourglass modes                */
    /*************************************************/
    CalcHourglassModes(xn,yn,zn,dvdxn,dvdyn,dvdzn,hourgam,volinv);

    /*************************************************/
    /*    CalcStressForElems                         */
    /*************************************************/
    Real_t B[3][8];

    CalcElemShapeFunctionDerivatives(xn, yn, zn, B, &det); 

    CalcElemNodeNormals( B[0] , B[1], B[2], xn, yn, zn); 

    // Check for bad volume
    if (det < 0.) {
      *bad_vol = elem; 
    }

    #pragma unroll
    for (int i=0;i<8;i++)
    {
      hgfx[i] = -( sigxx*B[0][i] );
      hgfy[i] = -( sigxx*B[1][i] );
      hgfz[i] = -( sigxx*B[2][i] );
    }

    if (hourg_gt_zero)
    {
      /*************************************************/
      /*    CalcFBHourglassForceForElems               */
      /*************************************************/


      #pragma unroll 
      for (int i=0;i<8;i++)
        xdn[i] =xd[n[i]];

      #pragma unroll 
      for (int i=0;i<8;i++)
        ydn[i] =yd[n[i]];

      #pragma unroll 
      for (int i=0;i<8;i++)
        zdn[i] =zd[n[i]];




      CalcElemFBHourglassForce
      ( &xdn[0],&ydn[0],&zdn[0],
	      hourgam[0],hourgam[1],hourgam[2],hourgam[3],
        hourgam[4],hourgam[5],hourgam[6],hourgam[7],
	    	coefficient,
	    	&hgfx[0],&hgfy[0],&hgfz[0]
      );

    }

#ifdef DOUBLE_PRECISION
    #pragma unroll
    for (int node=0;node<8;node++)
    {
      Index_t store_loc = elem+padded_numElem*node;
      fx_elem[store_loc]=hgfx[node]; 
      fy_elem[store_loc]=hgfy[node]; 
      fz_elem[store_loc]=hgfz[node];
    }
#else
    #pragma unroll
    for (int i=0;i<8;i++)
    {
      Index_t ni= n[i];
      atomicAdd(&fx_node[ni],hgfx[i]); 
      atomicAdd(&fy_node[ni],hgfy[i]); 
      atomicAdd(&fz_node[ni],hgfz[i]);
    } 
#endif

  } // If elem < numElem
}




template< bool hourg_gt_zero, int cta_size> 
__global__
void CalcVolumeForceForElems_kernel_warp_per_4cell(

    const Real_t* __restrict__ volo, 
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p, 
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss, 
    const Real_t* __restrict__ elemMass,
    //const Real_t __restrict__ *x,  const Real_t __restrict__ *y,  const Real_t __restrict__ *z,
    //const Real_t __restrict__ *xd,  const Real_t __restrict__ *yd,  const Real_t __restrict__ *zd,
    const Real_t  *x,  const Real_t *y,  const Real_t *z,
    const Real_t  *xd,  const Real_t *yd,  const Real_t *zd,
#ifdef DOUBLE_PRECISION // For floats, use atomicAdd
    Real_t* __restrict__ fx_elem, 
    Real_t* __restrict__ fy_elem, 
    Real_t* __restrict__ fz_elem,
#else
    Real_t* __restrict__ fx_node, 
    Real_t* __restrict__ fy_node, 
    Real_t* __restrict__ fz_node,
#endif
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads)

{

  /*************************************************
  *     FUNCTION: Calculates the volume forces
  *************************************************/

  Real_t xn,yn,zn;;
  Real_t xdn,ydn,zdn;;
  Real_t dvdxn,dvdyn,dvdzn;;
  Real_t hgfx,hgfy,hgfz;;
  Real_t hourgam[4];
  Real_t coefficient;

  int tid=blockDim.x*blockIdx.x+threadIdx.x;
  int elem = tid >> 3; // elem = tid/8
  int node = tid & 7; // node = tid%8


  if (elem < num_threads) 
  {
    Real_t volume = v[elem];
    Real_t det = volo[elem] * volume;

    // Check for bad volume
    if (volume < 0.) {
      *bad_vol = elem; 
    }

    Real_t ss1 = ss[elem];
    Real_t mass1 = elemMass[elem];
    Real_t sigxx = -p[elem] - q[elem];

    Index_t node_id;
    node_id = nodelist[elem+node*padded_numElem];

    Real_t volinv = Real_t(1.0) / det;

    xn =x[node_id];
    yn =y[node_id];
    zn =z[node_id];

    Real_t volume13 = CBRT(det);
    coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    /*************************************************/
    /*    compute the volume derivatives             */
    /*************************************************/
    unsigned int ind0,ind1,ind2,ind3,ind4,ind5;

    // Use octal number to represent the indices for each node
    //ind0 = 012307456;
    //ind1 = 023016745;
    //ind2 = 030125674;
    //ind3 = 045670123;
    //ind4 = 056743012;
    //ind5 = 074561230;

    //int mask = 7u << (3*node;

    switch(node) {
    case 0:
      {ind0=1; ind1=2; ind2=3; ind3=4; ind4=5; ind5=7;
      break;}
    case 1:
      {ind0=2; ind1=3; ind2=0; ind3=5; ind4=6; ind5=4;
      break;}
    case 2:
      {ind0=3; ind1=0; ind2=1; ind3=6; ind4=7; ind5=5;
      break;}
    case 3:
      {ind0=0; ind1=1; ind2=2; ind3=7; ind4=4; ind5=6;
      break;}
    case 4:
      {ind0=7; ind1=6; ind2=5; ind3=0; ind4=3; ind5=1;
      break;}
    case 5:
      {ind0=4; ind1=7; ind2=6; ind3=1; ind4=0; ind5=2;
      break;}
    case 6:
      {ind0=5; ind1=4; ind2=7; ind3=2; ind4=1; ind5=3;
      break;}
    case 7:
      {ind0=6; ind1=5; ind2=4; ind3=3; ind4=2; ind5=0;
      break;}
    }

    VOLUDER(utils::shfl(yn,ind0,8),utils::shfl(yn,ind1,8),utils::shfl(yn,ind2,8),
            utils::shfl(yn,ind3,8),utils::shfl(yn,ind4,8),utils::shfl(yn,ind5,8),
            utils::shfl(zn,ind0,8),utils::shfl(zn,ind1,8),utils::shfl(zn,ind2,8),
            utils::shfl(zn,ind3,8),utils::shfl(zn,ind4,8),utils::shfl(zn,ind5,8),
	          dvdxn);

    VOLUDER(utils::shfl(zn,ind0,8),utils::shfl(zn,ind1,8),utils::shfl(zn,ind2,8),
            utils::shfl(zn,ind3,8),utils::shfl(zn,ind4,8),utils::shfl(zn,ind5,8),
            utils::shfl(xn,ind0,8),utils::shfl(xn,ind1,8),utils::shfl(xn,ind2,8),
            utils::shfl(xn,ind3,8),utils::shfl(xn,ind4,8),utils::shfl(xn,ind5,8),
	          dvdyn);

    VOLUDER(utils::shfl(xn,ind0,8),utils::shfl(xn,ind1,8),utils::shfl(xn,ind2,8),
            utils::shfl(xn,ind3,8),utils::shfl(xn,ind4,8),utils::shfl(xn,ind5,8),
            utils::shfl(yn,ind0,8),utils::shfl(yn,ind1,8),utils::shfl(yn,ind2,8),
            utils::shfl(yn,ind3,8),utils::shfl(yn,ind4,8),utils::shfl(yn,ind5,8),
	          dvdzn);

    /*************************************************/
    /*    compute the hourglass modes                */
    /*************************************************/

    Real_t hourmodx, hourmody, hourmodz;
    const Real_t posf = Real_t( 1.);
    const Real_t negf = Real_t(-1.);

    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==2 || node==3 || node==4 || node==5) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[0] = negf;
    }
    else hourgam[0] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[0] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==2 || node==4 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[1] = negf;
    }
    else hourgam[1] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[1] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==3 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[2] = negf;
    }
    else hourgam[2] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[2] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==0 || node==2 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[3] = negf;
    }
    else hourgam[3] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[3] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    /*************************************************/
    /*    CalcStressForElems                         */
    /*************************************************/
    Real_t b[3];

    /*************************************************/
    //CalcElemShapeFunctionDerivatives_warp_per_4cell(xn, yn, zn, B, &det); 
    /*************************************************/

    Real_t fjxxi, fjxet, fjxze;
    Real_t fjyxi, fjyet, fjyze;
    Real_t fjzxi, fjzet, fjzze;

    fjxxi = fjxet = fjxze = Real_t(0.125)*xn;
    fjyxi = fjyet = fjyze = Real_t(0.125)*yn;
    fjzxi = fjzet = fjzze = Real_t(0.125)*zn;

    if (node==0 || node==3 || node==7 || node==4)
    {
      fjxxi = -fjxxi;
      fjyxi = -fjyxi;
      fjzxi = -fjzxi;
    }
    if (node==0 || node==5 || node==1 || node==4)
    {
      fjxet = -fjxet;
      fjyet = -fjyet;
      fjzet = -fjzet;
    }
    if (node==0 || node==3 || node==1 || node==2)
    {
      fjxze = -fjxze;
      fjyze = -fjyze;
      fjzze = -fjzze;
    }

    SumOverNodesShfl(fjxxi); 
    SumOverNodesShfl(fjxet); 
    SumOverNodesShfl(fjxze); 

    SumOverNodesShfl(fjyxi); 
    SumOverNodesShfl(fjyet); 
    SumOverNodesShfl(fjyze); 

    SumOverNodesShfl(fjzxi); 
    SumOverNodesShfl(fjzet); 
    SumOverNodesShfl(fjzze); 

    /* compute cofactors */
    Real_t cjxxi, cjxet, cjxze;
    Real_t cjyxi, cjyet, cjyze;
    Real_t cjzxi, cjzet, cjzze;

    cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
    cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
    cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

    cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
    cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
    cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

    cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
    cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
    cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

    Real_t coef_xi, coef_et, coef_ze;

    if (node==0 || node==3 || node==4 || node==7)
      coef_xi = Real_t(-1.);
    else
      coef_xi = Real_t(1.);

    if (node==0 || node==1 || node==4 || node==5) 
      coef_et = Real_t(-1.);
    else
      coef_et = Real_t(1.);

    if (node==0 || node==1 || node==2 || node==3) 
      coef_ze = Real_t(-1.);
    else
      coef_ze = Real_t(1.);

    /* calculate partials :
       this need only be done for l = 0,1,2,3   since , by symmetry ,
       (6,7,4,5) = - (0,1,2,3) .
    */
    b[0] =     coef_xi * cjxxi  +  coef_et * cjxet  +  coef_ze * cjxze;
    b[1] =     coef_xi * cjyxi  +  coef_et * cjyet  +  coef_ze * cjyze;
    b[2] =     coef_xi * cjzxi  +  coef_et * cjzet  +  coef_ze * cjzze;

    /* calculate jacobian determinant (volume) */
    det = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);

    /*************************************************/
    //CalcElemNodeNormals_warp_per_4cell( B[0] , B[1], B[2], xn, yn, zn); 
    /*************************************************/

    b[0] = Real_t(0.0);
    b[1] = Real_t(0.0);
    b[2] = Real_t(0.0);

    // Six faces, if no 
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 0,1,2,3);
                  
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 0,4,5,1);
 
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 1,5,6,2);
    
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 2,6,7,3);

    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 3,7,4,0);

    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 4,7,6,5);

    // Check for bad volume
    if (det < 0.) {
      *bad_vol = elem; 
    }

    hgfx = -( sigxx*b[0] );
    hgfy = -( sigxx*b[1] );
    hgfz = -( sigxx*b[2] );

    if (hourg_gt_zero)
    {
      /*************************************************/
      /*    CalcFBHourglassForceForElems               */
      /*************************************************/

      xdn = xd[node_id];
      ydn = yd[node_id];
      zdn = zd[node_id];

      Real_t hgfx_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*xdn;
          SumOverNodesShfl(h);
          hgfx_temp+=hourgam[i]*h;
      }
      hgfx_temp *= coefficient;
      hgfx += hgfx_temp;

      Real_t hgfy_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*ydn;
          SumOverNodesShfl(h);
          hgfy_temp+=hourgam[i]*h;
      }
      hgfy_temp *= coefficient;
      hgfy += hgfy_temp;

      Real_t hgfz_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*zdn;
          SumOverNodesShfl(h);
          hgfz_temp+=hourgam[i]*h;
      }
      hgfz_temp *= coefficient;
      hgfz += hgfz_temp;


    }

#ifdef DOUBLE_PRECISION
    Index_t store_loc = elem+padded_numElem*node;
    fx_elem[store_loc]=hgfx; 
    fy_elem[store_loc]=hgfy; 
    fz_elem[store_loc]=hgfz;
#else
    atomicAdd(&fx_node[node_id],hgfx); 
    atomicAdd(&fy_node[node_id],hgfy); 
    atomicAdd(&fz_node[node_id],hgfz);
#endif

  } // If elem < numElem
}
__global__
void CalcAccelerationForNodes_kernel(int numNode,
                                     Real_t *xdd, Real_t *ydd, Real_t *zdd,
                                     Real_t *fx, Real_t *fy, Real_t *fz,
                                     Real_t *nodalMass)
{
  int tid=blockDim.x*blockIdx.x+threadIdx.x;
  if (tid < numNode)
  {
      Real_t one_over_nMass = Real_t(1.)/nodalMass[tid];
      xdd[tid]=fx[tid]*one_over_nMass;
      ydd[tid]=fy[tid]*one_over_nMass;
      zdd[tid]=fz[tid]*one_over_nMass;
  }
}

__global__
void ApplyAccelerationBoundaryConditionsForNodes_kernel(
    int numNodeBC, Real_t *xyzdd, 
    Index_t *symm)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i < numNodeBC)
    {
          xyzdd[symm[i]] = Real_t(0.0) ;
    }
}


__device__
static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1); 
   Real_t temp = (fx * gx + fy * gy + fz * gz);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      temp * temp;
   return area ;
}

__device__
static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ; // 38
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = FMAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

__device__
static 
__forceinline__
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];
 
  Real_t tmp1 = (xvel[0]-xvel[6]);
  Real_t tmp2 = (xvel[1]-xvel[7]);
  Real_t tmp3 = (xvel[2]-xvel[4]);
  Real_t tmp4 = (xvel[3]-xvel[5]);


  d[0] = inv_detJ *  ( pfx[0] * tmp1
                     + pfx[1] * tmp2
                     + pfx[2] * tmp3
                     + pfx[3] * tmp4);

  dxddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  dxddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (yvel[0]-yvel[6]);
  tmp2 = (yvel[1]-yvel[7]);
  tmp3 = (yvel[2]-yvel[4]);
  tmp4 = (yvel[3]-yvel[5]);

  d[1] = inv_detJ *  ( pfy[0] * tmp1
                     + pfy[1] * tmp2
                     + pfy[2] * tmp3
                     + pfy[3] * tmp4);

  dyddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dyddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (zvel[0]-zvel[6]);
  tmp2 = (zvel[1]-zvel[7]);
  tmp3 = (zvel[2]-zvel[4]);
  tmp4 = (zvel[3]-zvel[5]);

  d[2] = inv_detJ * ( pfz[0] * tmp1
                     + pfz[1] * tmp2
                     + pfz[2] * tmp3
                     + pfz[3] * tmp4);

  dzddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dzddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

static __device__ __forceinline__
void CalcMonoGradient(Real_t *x, Real_t *y, Real_t *z,
                      Real_t *xv, Real_t *yv, Real_t *zv,
                      Real_t vol, 
                      Real_t *delx_zeta, 
                      Real_t *delv_zeta,
                      Real_t *delx_xi,
                      Real_t *delv_xi,
                      Real_t *delx_eta,
                      Real_t *delv_eta)
{

   #define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = Real_t(1.e-36) ;
   Real_t ax,ay,az ;
   Real_t dxv,dyv,dzv ;

   Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

   Real_t dxj = Real_t(-0.25)*(SUM4(x[0],x[1],x[5],x[4]) - SUM4(x[3],x[2],x[6],x[7])) ;
   Real_t dyj = Real_t(-0.25)*(SUM4(y[0],y[1],y[5],y[4]) - SUM4(y[3],y[2],y[6],y[7])) ;
   Real_t dzj = Real_t(-0.25)*(SUM4(z[0],z[1],z[5],z[4]) - SUM4(z[3],z[2],z[6],z[7])) ;

   Real_t dxi = Real_t( 0.25)*(SUM4(x[1],x[2],x[6],x[5]) - SUM4(x[0],x[3],x[7],x[4])) ;
   Real_t dyi = Real_t( 0.25)*(SUM4(y[1],y[2],y[6],y[5]) - SUM4(y[0],y[3],y[7],y[4])) ;
   Real_t dzi = Real_t( 0.25)*(SUM4(z[1],z[2],z[6],z[5]) - SUM4(z[0],z[3],z[7],z[4])) ;

   Real_t dxk = Real_t( 0.25)*(SUM4(x[4],x[5],x[6],x[7]) - SUM4(x[0],x[1],x[2],x[3])) ;
   Real_t dyk = Real_t( 0.25)*(SUM4(y[4],y[5],y[6],y[7]) - SUM4(y[0],y[1],y[2],y[3])) ;
   Real_t dzk = Real_t( 0.25)*(SUM4(z[4],z[5],z[6],z[7]) - SUM4(z[0],z[1],z[2],z[3])) ;

   /* find delvk and delxk ( i cross j ) */
   ax = dyi*dzj - dzi*dyj ;
   ay = dzi*dxj - dxi*dzj ;
   az = dxi*dyj - dyi*dxj ;

   *delx_zeta = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ; 

   ax *= norm ;
   ay *= norm ;
   az *= norm ; 

   dxv = Real_t(0.25)*(SUM4(xv[4],xv[5],xv[6],xv[7]) - SUM4(xv[0],xv[1],xv[2],xv[3])) ;
   dyv = Real_t(0.25)*(SUM4(yv[4],yv[5],yv[6],yv[7]) - SUM4(yv[0],yv[1],yv[2],yv[3])) ;
   dzv = Real_t(0.25)*(SUM4(zv[4],zv[5],zv[6],zv[7]) - SUM4(zv[0],zv[1],zv[2],zv[3])) ; 

   *delv_zeta = ax*dxv + ay*dyv + az*dzv ;

   /* find delxi and delvi ( j cross k ) */

   ax = dyj*dzk - dzj*dyk ;
   ay = dzj*dxk - dxj*dzk ;
   az = dxj*dyk - dyj*dxk ;

   *delx_xi = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(0.25)*(SUM4(xv[1],xv[2],xv[6],xv[5]) - SUM4(xv[0],xv[3],xv[7],xv[4])) ;
   dyv = Real_t(0.25)*(SUM4(yv[1],yv[2],yv[6],yv[5]) - SUM4(yv[0],yv[3],yv[7],yv[4])) ;
   dzv = Real_t(0.25)*(SUM4(zv[1],zv[2],zv[6],zv[5]) - SUM4(zv[0],zv[3],zv[7],zv[4])) ;

   *delv_xi = ax*dxv + ay*dyv + az*dzv ;

   /* find delxj and delvj ( k cross i ) */

   ax = dyk*dzi - dzk*dyi ;
   ay = dzk*dxi - dxk*dzi ;
   az = dxk*dyi - dyk*dxi ;

   *delx_eta = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(-0.25)*(SUM4(xv[0],xv[1],xv[5],xv[4]) - SUM4(xv[3],xv[2],xv[6],xv[7])) ;
   dyv = Real_t(-0.25)*(SUM4(yv[0],yv[1],yv[5],yv[4]) - SUM4(yv[3],yv[2],yv[6],yv[7])) ;
   dzv = Real_t(-0.25)*(SUM4(zv[0],zv[1],zv[5],zv[4]) - SUM4(zv[3],zv[2],zv[6],zv[7])) ;

   *delv_eta = ax*dxv + ay*dyv + az*dzv ;
#undef SUM4
}


__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(64,8) // 64-bit
#else
__launch_bounds__(64,16) // 32-bit
#endif
void CalcKinematicsAndMonotonicQGradient_kernel(
    Index_t numElem, Index_t padded_numElem, const Real_t dt,
    const Index_t* __restrict__ nodelist, const Real_t* __restrict__ volo, const Real_t* __restrict__ v,

    const Real_t* __restrict__ x, 
    const Real_t* __restrict__ y, 
    const Real_t* __restrict__ z,
    const Real_t* __restrict__ xd, 
    const Real_t* __restrict__ yd, 
    const Real_t* __restrict__ zd,

    //TextureObj<Real_t> x, 
    //TextureObj<Real_t> y, 
    //TextureObj<Real_t> z,
    //TextureObj<Real_t> xd, 
    //TextureObj<Real_t> yd, 
    //TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x, 
    //TextureObj<Real_t>* y, 
    //TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd, 
    //TextureObj<Real_t>* yd, 
    //TextureObj<Real_t>* zd,
 

    Real_t* __restrict__ vnew,
    Real_t* __restrict__ delv,
    Real_t* __restrict__ arealg,
    Real_t* __restrict__ dxx,
    Real_t* __restrict__ dyy,
    Real_t* __restrict__ dzz,
    Real_t* __restrict__ vdov,
    Real_t* __restrict__ delx_zeta, 
    Real_t* __restrict__ delv_zeta,
    Real_t* __restrict__ delx_xi, 
    Real_t* __restrict__ delv_xi, 
    Real_t* __restrict__ delx_eta,
    Real_t* __restrict__ delv_eta,
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads
    )
{

  Real_t B[3][8] ; /** shape function derivatives */
  Index_t nodes[8] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t D[6];

  int k=blockDim.x*blockIdx.x+threadIdx.x;

  if ( k < num_threads) {

    Real_t volume ;
    Real_t relativeVolume ;

    // get nodal coordinates from global arrays and copy into local arrays.
    //#pragma unroll
    //for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    //{
    //  Index_t gnode = nodelist[k+lnode*padded_numElem];
    //  nodes[lnode] = gnode;
    //  x_local[lnode] = x[gnode];
    //  y_local[lnode] = y[gnode];
    //  z_local[lnode] = z[gnode];
    //}

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*padded_numElem];
      nodes[lnode] = gnode;
    }

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      x_local[lnode] = x[nodes[lnode]];

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      y_local[lnode] = y[nodes[lnode]];

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      z_local[lnode] = z[nodes[lnode]];



    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local ); 

    relativeVolume = volume / volo[k] ; 
    vnew[k] = relativeVolume ;

    delv[k] = relativeVolume - v[k] ;
    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local,y_local,z_local,volume);

    // get nodal velocities from global array and copy into local arrays.
    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodes[lnode];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t dt2 = Real_t(0.5) * dt;

    #pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j]; 
    }

    Real_t detJ;

    CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ );

    CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);

    // ------------------------
    // CALC LAGRANGE ELEM 2
    // ------------------------

    // calc strain rate and apply as constraint (only done in FB element)
    Real_t vdovNew = D[0] + D[1] + D[2];
    Real_t vdovthird = vdovNew/Real_t(3.0) ;
    
    // make the rate of deformation tensor deviatoric
    vdov[k] = vdovNew ;
    dxx[k] = D[0] - vdovthird ;
    dyy[k] = D[1] - vdovthird ;
    dzz[k] = D[2] - vdovthird ; 

    // ------------------------
    // CALC MONOTONIC Q GRADIENT
    // ------------------------
    Real_t vol = volo[k]*vnew[k];

   // Undo x_local update
    #pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j ) {
       x_local[j] += dt2 * xd_local[j];
       y_local[j] += dt2 * yd_local[j];
       z_local[j] += dt2 * zd_local[j]; 
    }

   CalcMonoGradient(x_local,y_local,z_local,xd_local,yd_local,zd_local,
                          vol, 
                          &delx_zeta[k],&delv_zeta[k],&delx_xi[k],
                          &delv_xi[k], &delx_eta[k], &delv_eta[k]);

  //Check for bad volume 
  if (relativeVolume < 0)
    *bad_vol = k;
  }
}

__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(128,16) 
#else
__launch_bounds__(128,16) 
#endif
void CalcMonotonicQRegionForElems_kernel(
    Real_t qlc_monoq,
    Real_t qqc_monoq,
    Real_t monoq_limiter_mult,
    Real_t monoq_max_slope,
    Real_t ptiny,
    
    Index_t elength,
  
    Index_t* regElemlist,  
    Index_t *elemBC,
    Index_t *lxim,
    Index_t *lxip,
    Index_t *letam,
    Index_t *letap,
    Index_t *lzetam,
    Index_t *lzetap,
    Real_t *delv_xi,
    Real_t *delv_eta,
    Real_t *delv_zeta,
    Real_t *delx_xi,
    Real_t *delx_eta,
    Real_t *delx_zeta,
    Real_t *vdov,Real_t *elemMass,Real_t *volo,Real_t *vnew,
    Real_t *qq, Real_t *ql,
    Real_t *q,
    Real_t qstop,
    Index_t* bad_q 
    )
{
    int ielem=blockDim.x*blockIdx.x + threadIdx.x;

    if (ielem<elength) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = regElemlist[ielem];
      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
	 case XI_M_COMM: /* needs comm data */
         case 0:         delvm = delv_xi[lxim[i]] ; break ;
         case XI_M_SYMM: delvm = delv_xi[i] ;            break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }
      switch (bcMask & XI_P) {
	 case XI_P_COMM: /* needs comm data */
         case 0:         delvp = delv_xi[lxip[i]] ; break ;
         case XI_P_SYMM: delvp = delv_xi[i] ;            break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
	 case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = delv_eta[letam[i]] ; break ;
         case ETA_M_SYMM: delvm = delv_eta[i] ;             break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }
      switch (bcMask & ETA_P) {
	 case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = delv_eta[letap[i]] ; break ;
         case ETA_P_SYMM: delvp = delv_eta[i] ;             break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
         case ZETA_M_SYMM: delvm = delv_zeta[i] ;              break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
         case ZETA_P_SYMM: delvp = delv_zeta[i] ;              break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( vdov[i] > Real_t(0.) ) {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
     }
      else {
         Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
         Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
         Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      qq[i] = qquad ;
      ql[i] = qlin  ;

      // Don't allow excessive artificial viscosity
      if (q[i] > qstop)
        *(bad_q) = i;

   }
}
static 
__device__ __forceinline__
void CalcPressureForElems_device(
                      Real_t& p_new, Real_t& bvc,
                      Real_t& pbvc, Real_t& e_old,
                      Real_t& compression, Real_t& vnewc,
                      Real_t pmin,
                      Real_t p_cut, Real_t eosvmax)
{
      
      Real_t c1s = Real_t(2.0)/Real_t(3.0); 
      Real_t p_temp = p_new;

      bvc = c1s * (compression + Real_t(1.));
      pbvc = c1s;

      p_temp = bvc * e_old ;

      if ( FABS(p_temp) <  p_cut )
        p_temp = Real_t(0.0) ;

      if ( vnewc >= eosvmax ) /* impossible condition here? */
        p_temp = Real_t(0.0) ;

      if (p_temp < pmin)
        p_temp = pmin ;

      p_new = p_temp;

}

static
__device__ __forceinline__
void CalcSoundSpeedForElems_device(Real_t& vnewc, Real_t rho0, Real_t &enewc,
                            Real_t &pnewc, Real_t &pbvc,
                            Real_t &bvc, Real_t ss4o3, Index_t nz,
                            Real_t *ss, Index_t iz)
{
  Real_t ssTmp = (pbvc * enewc + vnewc * vnewc *
             bvc * pnewc) / rho0;
  if (ssTmp <= Real_t(.1111111e-36)) {
     ssTmp = Real_t(.3333333e-18);
  }
  else {
    ssTmp = SQRT(ssTmp) ;
  }
  ss[iz] = ssTmp;
}
static
__device__
__forceinline__ 
void ApplyMaterialPropertiesForElems_device(
    Real_t& eosvmin, Real_t& eosvmax,
    Real_t* vnew, Real_t *v,
    Real_t& vnewc, Index_t* bad_vol, Index_t zn)
{
  vnewc = vnew[zn] ;

  if (eosvmin != Real_t(0.)) {
      if (vnewc < eosvmin)
          vnewc = eosvmin ;
  }

  if (eosvmax != Real_t(0.)) {
      if (vnewc > eosvmax)
          vnewc = eosvmax ;
  }

  // Now check for valid volume
  Real_t vc = v[zn];
  if (eosvmin != Real_t(0.)) {
     if (vc < eosvmin)
        vc = eosvmin ;
  }
  if (eosvmax != Real_t(0.)) {
     if (vc > eosvmax)
        vc = eosvmax ;
  }
  if (vc <= 0.) {
     *bad_vol = zn;
  }

}

static
__device__
__forceinline__
void UpdateVolumesForElems_device(Index_t numElem, Real_t& v_cut,
                                  Real_t *vnew,
                                  Real_t *v,
                                  int i)
{
   Real_t tmpV ;
   tmpV = vnew[i] ;

   if ( FABS(tmpV - Real_t(1.0)) < v_cut )
      tmpV = Real_t(1.0) ;
   v[i] = tmpV ;
}


static
__device__
__forceinline__
void CalcEnergyForElems_device(Real_t& p_new, Real_t& e_new, Real_t& q_new,
                            Real_t& bvc, Real_t& pbvc,
                            Real_t& p_old, Real_t& e_old, Real_t& q_old,
                            Real_t& compression, Real_t& compHalfStep,
                            Real_t& vnewc, Real_t& work, Real_t& delvc, Real_t pmin,
                            Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin,
                            Real_t& qq, Real_t& ql,
                            Real_t& rho0,
                            Real_t& eosvmax,
                            Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_t pHalfStep;

   e_new = e_old - Real_t(0.5) * delvc * (p_old + q_old)
      + Real_t(0.5) * work;

   if (e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax);

   Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep) ;

   if ( delvc > Real_t(0.) ) {
      q_new = Real_t(0.) ;
   }
   else {
      Real_t ssc = ( pbvc * e_new
              + vhalf * vhalf * bvc * pHalfStep ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc =Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_new = (ssc*ql + qq) ;
   }

   e_new = e_new + Real_t(0.5) * delvc
      * (  Real_t(3.0)*(p_old     + q_old)
           - Real_t(4.0)*(pHalfStep + q_new)) ;

   e_new += Real_t(0.5) * work;

   if (FABS(e_new) < e_cut) {
      e_new = Real_t(0.)  ;
   }
   if (     e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax);

   Real_t q_tilde ;

   if (delvc > Real_t(0.)) {
      q_tilde = Real_t(0.) ;
   }
   else {
      Real_t ssc = ( pbvc * e_new
              + vnewc * vnewc * bvc * p_new ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc = Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_tilde = (ssc*ql + qq) ;
   }

   e_new = e_new - (  Real_t(7.0)*(p_old     + q_old)
                            - Real_t(8.0)*(pHalfStep + q_new)
                            + (p_new + q_tilde)) * delvc*sixth ;

   if (FABS(e_new) < e_cut) {
      e_new = Real_t(0.)  ;
   }
   if ( e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax);


   if ( delvc <= Real_t(0.) ) {
      Real_t ssc = ( pbvc * e_new
              + vnewc * vnewc * bvc * p_new ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc = Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_new = (ssc*ql + qq) ;

      if (FABS(q_new) < q_cut) q_new = Real_t(0.) ;
   }

   return ;
}

__device__ inline
Index_t giveMyRegion(const Index_t* regCSR,const Index_t i, const Index_t numReg)
{

  for(Index_t reg = 0; reg < numReg-1; reg++)
	if(i < regCSR[reg])
		return reg;
  return (numReg-1);
}


__global__
void ApplyMaterialPropertiesAndUpdateVolume_kernel(
        Index_t length,
        Real_t rho0,
        Real_t e_cut,
        Real_t emin,
        Real_t* __restrict__ ql,
        Real_t* __restrict__ qq,
        Real_t* __restrict__ vnew,
        Real_t* __restrict__ v,
        Real_t pmin,
        Real_t p_cut,
        Real_t q_cut,
        Real_t eosvmin,
        Real_t eosvmax,
        Index_t* __restrict__ regElemlist,
        Real_t* __restrict__ e,
        Real_t* __restrict__ delv,
        Real_t* __restrict__ p,
        Real_t* __restrict__ q,
        Real_t ss4o3,
        Real_t* __restrict__ ss,
        Real_t v_cut,
        Index_t* __restrict__ bad_vol, 
        const Int_t cost,
        const Index_t* regCSR,
        const Index_t* regReps,
	const Index_t  numReg
)
{

  Real_t e_old, delvc, p_old, q_old, e_temp, delvc_temp, p_temp, q_temp;
  Real_t compression, compHalfStep;
  Real_t qq_old, ql_old, qq_temp, ql_temp, work;
  Real_t p_new, e_new, q_new;
  Real_t bvc, pbvc, vnewc;

  Index_t i=blockDim.x*blockIdx.x + threadIdx.x;

  if (i<length) {

    Index_t zidx  = regElemlist[i] ;

    ApplyMaterialPropertiesForElems_device
      (eosvmin,eosvmax,vnew,v,vnewc,bad_vol,zidx);

  Index_t region = giveMyRegion(regCSR,i,numReg);  
  Index_t rep = regReps[region];
    
    e_temp = e[zidx];
    p_temp = p[zidx];
    q_temp = q[zidx];
    qq_temp = qq[zidx] ;
    ql_temp = ql[zidx] ;
    delvc_temp = delv[zidx];

  for(int r=0; r < rep; r++)
  {

    e_old = e_temp;
    p_old = p_temp;
    q_old = q_temp;
    qq_old = qq_temp;
    ql_old = ql_temp;
    delvc = delvc_temp;
    work = Real_t(0.);

    Real_t vchalf ;
    compression = Real_t(1.) / vnewc - Real_t(1.);
    vchalf = vnewc - delvc * Real_t(.5);
    compHalfStep = Real_t(1.) / vchalf - Real_t(1.);

    if ( eosvmin != Real_t(0.) ) {
        if (vnewc <= eosvmin) { /* impossible due to calling func? */
            compHalfStep = compression ;
        }
    }
    if ( eosvmax != Real_t(0.) ) {
        if (vnewc >= eosvmax) { /* impossible due to calling func? */
            p_old        = Real_t(0.) ;
            compression  = Real_t(0.) ;
            compHalfStep = Real_t(0.) ;
        }
    }



    CalcEnergyForElems_device(p_new, e_new, q_new, bvc, pbvc,
                 p_old, e_old,  q_old, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 qq_old, ql_old, rho0, eosvmax, length);
 }

    p[zidx] = p_new ;
    e[zidx] = e_new ;
    q[zidx] = q_new ;

    CalcSoundSpeedForElems_device
       (vnewc,rho0,e_new,p_new,pbvc,bvc,ss4o3,length,ss,zidx);


    UpdateVolumesForElems_device(length,v_cut,vnew,v,zidx);

  }
}


/*
 * CalcTimeConstraintsForElems_kernel - CUDA kernel to calculate timestep constraints
 * 
 * This kernel computes the maximum allowable timestep for the next iteration
 * based on two constraints:
 * 1. The Courant constraint (mindtcourant) - based on element sound speed and size
 * 2. The volume change constraint (mindthydro) - limits excessive element deformation
 *
 * Each thread calculates constraints for one element, then a parallel reduction
 * finds the minimum values across all elements in the block.
 *
 * Template parameter:
 *   block_size - Number of threads per block, must match the launch configuration
 */
template<int block_size>
__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(128,16)  // Optimize register usage for double precision
#else
__launch_bounds__(128,16)  // Optimize register usage for single precision
#endif
void CalcTimeConstraintsForElems_kernel(
    Index_t length,        // Total number of elements to process
    Real_t qqc2,           // Artificial viscosity coefficient squared
    Real_t dvovmax,        // Maximum allowable volume change ratio
    Index_t *matElemlist,  // List of material elements to process
    Real_t *ss,            // Sound speed array
    Real_t *vdov,          // Volume derivative over volume
    Real_t *arealg,        // Element characteristic length
    Real_t *dev_mindtcourant,  // Output: minimum courant timestep per block
    Real_t *dev_mindthydro)    // Output: minimum hydro timestep per block
{
    // Thread ID within block
    int tid = threadIdx.x;
    // Global thread ID (unique element index)
    int i = blockDim.x*blockIdx.x + tid;

    // Shared memory for block-level reductions
    // These arrays store intermediate results during min-finding
    __shared__ volatile Real_t s_mindthydro[block_size];
    __shared__ volatile Real_t s_mindtcourant[block_size];

    // Initialize to very large values (effectively infinity)
    Real_t mindthydro = Real_t(1.0e+20);
    Real_t mindtcourant = Real_t(1.0e+20);

    // Local working variable for hydro timestep
    Real_t dthydro = mindthydro;
    Real_t dtcourant = mindtcourant;

    while (i<length) {

      Index_t indx = matElemlist[i] ;
      Real_t vdov_tmp = vdov[indx];

      // Computing dt_hydro
      if (vdov_tmp != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(vdov_tmp)+Real_t(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
         }
      }
      if (dthydro < mindthydro)
        mindthydro = dthydro;

      // Computing dt_courant
      Real_t ss_tmp = ss[indx];
      Real_t area_tmp = arealg[indx];
      Real_t dtf = ss_tmp * ss_tmp ;
    
      dtf += ((vdov_tmp < 0.) ? qqc2*area_tmp*area_tmp*vdov_tmp*vdov_tmp : 0.);

      dtf = area_tmp / SQRT(dtf) ;

      /* determine minimum timestep with its corresponding elem */
      if (vdov_tmp != Real_t(0.) && dtf < dtcourant) {
        dtcourant = dtf ;
      }

      if (dtcourant< mindtcourant)
        mindtcourant= dtcourant;

      i += gridDim.x*blockDim.x;
    }

    s_mindthydro[tid]   = mindthydro;
    s_mindtcourant[tid] = mindtcourant;

    __syncthreads();

    // Do shared memory reduction
    if (block_size >= 1024) { 
      if (tid < 512) { 
        s_mindthydro[tid]   = min( s_mindthydro[tid]  , s_mindthydro[tid + 512]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 512]) ; }
      __syncthreads();  }

    if (block_size >=  512) { 
      if (tid < 256) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid + 256]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 256]) ; } 
      __syncthreads(); }

    if (block_size >=  256) { 
      if (tid < 128) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid + 128]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 128]) ; } 
      __syncthreads(); }

    if (block_size >=  128) { 
      if (tid <  64) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  64]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  64]) ; } 
      __syncthreads(); }

    if (tid <  32) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  32]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  32]) ; 
    } 

    if (tid <  16) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  16]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  16]) ;  
    } 
    if (tid <   8) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   8]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   8]) ;  
    } 
    if (tid <   4) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   4]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   4]) ;  
    } 
    if (tid <   2) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   2]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   2]) ;  
    } 
    if (tid <   1) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   1]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   1]) ;  
    } 

    // Store in global memory
    if (tid==0) {
      dev_mindtcourant[blockIdx.x] = s_mindtcourant[0];
      dev_mindthydro[blockIdx.x] = s_mindthydro[0];
    }

}

template <int block_size>
__global__ 
void CalcMinDtOneBlock(Real_t* dev_mindthydro, Real_t* dev_mindtcourant, Real_t* dtcourant, Real_t* dthydro, Index_t shared_array_size)
{

  volatile __shared__ Real_t s_data[block_size];
  int tid = threadIdx.x;

  if (blockIdx.x==0)
  {
    if (tid < shared_array_size)
      s_data[tid] = dev_mindtcourant[tid];
    else
      s_data[tid] = 1.0e20;

    __syncthreads();

    if (block_size >= 1024) { if (tid < 512) { s_data[tid] = min(s_data[tid],s_data[tid + 512]); } __syncthreads(); }
    if (block_size >=  512) { if (tid < 256) { s_data[tid] = min(s_data[tid],s_data[tid + 256]); } __syncthreads(); }
    if (block_size >=  256) { if (tid < 128) { s_data[tid] = min(s_data[tid],s_data[tid + 128]); } __syncthreads(); }
    if (block_size >=  128) { if (tid <  64) { s_data[tid] = min(s_data[tid],s_data[tid +  64]); } __syncthreads(); }
    if (tid <  32) { s_data[tid] = min(s_data[tid],s_data[tid +  32]); } 
    if (tid <  16) { s_data[tid] = min(s_data[tid],s_data[tid +  16]); } 
    if (tid <   8) { s_data[tid] = min(s_data[tid],s_data[tid +   8]); } 
    if (tid <   4) { s_data[tid] = min(s_data[tid],s_data[tid +   4]); } 
    if (tid <   2) { s_data[tid] = min(s_data[tid],s_data[tid +   2]); } 
    if (tid <   1) { s_data[tid] = min(s_data[tid],s_data[tid +   1]); } 

    if (tid<1)
    {
      *(dtcourant)= s_data[0];
    }
  }
  else if (blockIdx.x==1)
  {
    if (tid < shared_array_size)
      s_data[tid] = dev_mindthydro[tid];
    else
      s_data[tid] = 1.0e20;

    __syncthreads();

    if (block_size >= 1024) { if (tid < 512) { s_data[tid] = min(s_data[tid],s_data[tid + 512]); } __syncthreads(); }
    if (block_size >=  512) { if (tid < 256) { s_data[tid] = min(s_data[tid],s_data[tid + 256]); } __syncthreads(); }
    if (block_size >=  256) { if (tid < 128) { s_data[tid] = min(s_data[tid],s_data[tid + 128]); } __syncthreads(); }
    if (block_size >=  128) { if (tid <  64) { s_data[tid] = min(s_data[tid],s_data[tid +  64]); } __syncthreads(); }
    if (tid <  32) { s_data[tid] = min(s_data[tid],s_data[tid +  32]); } 
    if (tid <  16) { s_data[tid] = min(s_data[tid],s_data[tid +  16]); } 
    if (tid <   8) { s_data[tid] = min(s_data[tid],s_data[tid +   8]); } 
    if (tid <   4) { s_data[tid] = min(s_data[tid],s_data[tid +   4]); } 
    if (tid <   2) { s_data[tid] = min(s_data[tid],s_data[tid +   2]); } 
    if (tid <   1) { s_data[tid] = min(s_data[tid],s_data[tid +   1]); } 

    if (tid<1)
    {
      *(dthydro) = s_data[0];
    }
  }
}
