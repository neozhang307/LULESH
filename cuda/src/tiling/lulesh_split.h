#ifndef LULESH_H
#define LULESH_H

#include "utility/vector.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define LULESH_SHOW_PROGRESS 0
#define DOUBLE_PRECISION
//#define SAMI 

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define POW(a, b) (pow((a), (b)))

/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

#define USE_TILING




enum {
  VolumeError = -1,
  QStopError = -2,
  LFileError = -3
} ;

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
typedef double       real8 ;

typedef int    Index_t ; /* array subscript and loop index */
typedef int    Int_t ;   /* integer representation */
#ifdef DOUBLE_PRECISION
typedef real8  Real_t ;  /* floating point representation */
#else
typedef real4  Real_t ;  /* floating point representation */
#endif

class Domain
{

public: 

  void sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h);
  void CreateRegionIndexSets(Int_t nr, Int_t balance, Int_t tildID);


  Index_t max_streams;
  std::vector<cudaStream_t> streams;

  /* Elem-centered */

  Index_t* matElemlist ; /* material indexset */
  Index_t* nodelist ;    /* elemToNode connectivity */

  Index_t* lxim ;        /* element connectivity through face */
  Index_t* lxip ;
  Index_t* letam ;
  Index_t* letap ;
  Index_t* lzetam ;
  Index_t* lzetap ;

  Int_t* elemBC ;        /* elem face symm/free-surf flag */

  Real_t* e ;            /* energy */

  Real_t* p ;            /* pressure */

  Real_t* q ;            /* q */
  Real_t* ql ;           /* linear term for q */
  Real_t* qq ;           /* quadratic term for q */

  Real_t* v ;            /* relative volume */

  Real_t* volo ;         /* reference volume */
  Real_t* delv ;         /* m_vnew - m_v */
  Real_t* vdov ;         /* volume derivative over volume */

  Real_t* arealg ;       /* char length of an element */
  
  Real_t* ss ;           /* "sound speed" */

  Real_t* elemMass ;     /* mass */
/******************************************************** */
  Real_t* vnew ;         /* new relative volume -- temporary */

  Real_t* delv_xi ;      /* velocity gradient -- temporary */
  Real_t* delv_eta ;
  Real_t* delv_zeta ;

  Real_t* delx_xi ;      /* coordinate gradient -- temporary */
  Real_t* delx_eta ;
  Real_t* delx_zeta ;

  Real_t* dxx ;          /* principal strains -- temporary */
  Real_t* dyy ;
  Real_t* dzz ;
/******************************************************** */
  /* Node-centered */

  Real_t* x ;            /* coordinates */
  Real_t* y ;
  Real_t* z ;

  Real_t* xd ;           /* velocities */
  Real_t* yd ;
  Real_t* zd ;


  Real_t* xdd ;          /* accelerations */
  Real_t* ydd ;
  Real_t* zdd ;

  Real_t* fx ;           /* forces */
  Real_t* fy ;
  Real_t* fz ;

  Real_t* nodalMass ;    /* mass */
  Real_t* h_nodalMass ;    /* mass - host */

  /* device pointers for comms */
  Real_t *d_delv_xi ;      /* velocity gradient -- temporary */
  Real_t *d_delv_eta ;
  Real_t *d_delv_zeta ;

  Real_t *d_x ;            /* coordinates */
  Real_t *d_y ;
  Real_t *d_z ;

  Real_t *d_xd ;           /* velocities */
  Real_t *d_yd ;
  Real_t *d_zd ;

  Real_t *d_fx ;           /* forces */
  Real_t *d_fy ;
  Real_t *d_fz ;

  /* access elements for comms */
  Real_t& get_delv_xi(Index_t idx) { return d_delv_xi[idx] ; }
  Real_t& get_delv_eta(Index_t idx) { return d_delv_eta[idx] ; }
  Real_t& get_delv_zeta(Index_t idx) { return d_delv_zeta[idx] ; }

  Real_t& get_x(Index_t idx) { return d_x[idx] ; }
  Real_t& get_y(Index_t idx) { return d_y[idx] ; }
  Real_t& get_z(Index_t idx) { return d_z[idx] ; }

  Real_t& get_xd(Index_t idx) { return d_xd[idx] ; }
  Real_t& get_yd(Index_t idx) { return d_yd[idx] ; }
  Real_t& get_zd(Index_t idx) { return d_zd[idx] ; }

  Real_t& get_fx(Index_t idx) { return d_fx[idx] ; }
  Real_t& get_fy(Index_t idx) { return d_fy[idx] ; }
  Real_t& get_fz(Index_t idx) { return d_fz[idx] ; }

  // host access
  Real_t& get_nodalMass(Index_t idx) { return h_nodalMass[idx] ; }

  /* Boundary nodesets */

  Index_t* symmX ;       /* symmetry plane nodesets */
  Index_t* symmY ;        
  Index_t* symmZ ;
   
  Int_t* nodeElemCount ;
  Int_t* nodeElemStart;
  Index_t* nodeElemCornerList ;

  /* Parameters */

  Real_t dtfixed ;               /* fixed time increment */
  Real_t deltatimemultlb ;
  Real_t deltatimemultub ;
  Real_t stoptime ;              /* end time for simulation */
  Real_t dtmax ;                 /* maximum allowable time increment */
  Int_t cycle ;                  /* iteration count for simulation */

  Real_t* dthydro_h;             /* hydro time constraint */ 
  Real_t* dtcourant_h;           /* courant time constraint */
  Index_t* bad_q_h;              /* flag to indicate Q error */
  Index_t* bad_vol_h;            /* flag to indicate volume error */

  /* cuda Events to indicate completion of certain kernels */
  cudaEvent_t time_constraint_computed;

  Real_t time_h ;               /* current time */
  Real_t deltatime_h ;          /* variable time increment */

  Real_t u_cut ;                /* velocity tolerance */
  Real_t hgcoef ;               /* hourglass control */
  Real_t qstop ;                /* excessive q indicator */
  Real_t monoq_max_slope ;
  Real_t monoq_limiter_mult ;   
  Real_t e_cut ;                /* energy tolerance */
  Real_t p_cut ;                /* pressure tolerance */
  Real_t ss4o3 ;
  Real_t q_cut ;                /* q tolerance */
  Real_t v_cut ;                /* relative volume tolerance */
  Real_t qlc_monoq ;            /* linear term coef for q */
  Real_t qqc_monoq ;            /* quadratic term coef for q */
  Real_t qqc ;
  Real_t eosvmax ;
  Real_t eosvmin ;
  Real_t pmin ;                 /* pressure floor */
  Real_t emin ;                 /* energy floor */
  Real_t dvovmax ;              /* maximum allowable volume change */
  Real_t refdens ;              /* reference density */

  Index_t m_colLoc ;
  Index_t m_rowLoc ;
  Index_t m_planeLoc ;
  Index_t m_tp ;

  Index_t&  colLoc()             { return m_colLoc ; }
  Index_t&  rowLoc()             { return m_rowLoc ; }
  Index_t&  planeLoc()           { return m_planeLoc ; }
  Index_t&  tp()                 { return m_tp ; }

  Index_t sizeX ;
  Index_t sizeY ;
  Index_t sizeZ ;
  Index_t maxPlaneSize ;
  Index_t maxEdgeSize ;

  Index_t numElem ;
  Index_t padded_numElem ; 

  Index_t numNode;
  Index_t padded_numNode ; 

  Index_t numSymmX ; 
  Index_t numSymmY ; 
  Index_t numSymmZ ; 

  Index_t octantCorner;

   // Region information
   Int_t numReg ; //number of regions (def:11)
   Int_t balance; //Load balance between regions of a domain (def: 1)
   Int_t  cost;  //imbalance cost (def: 1)
   Int_t*   regElemSize ;   // Size of region sets
   Int_t* regCSR;  // records the begining and end of each region
   Int_t* regReps; // records the rep number per region
   Index_t* regNumList;    // Region number per domain element
   Index_t* regElemlist;  // region indexset 
   Index_t* regSorted; // keeps index of sorted regions
   
   //
   // MPI-Related additional data
   //

   Index_t m_numRanks;
   Index_t& numRanks() { return m_numRanks ; }

   void SetupCommBuffers(Int_t edgeNodes);
   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Int_t domNodes, Int_t padded_domElems, Vector_h<Real_t> &x_h, Vector_h<Real_t> &y_h, Vector_h<Real_t> &z_h, Vector_h<Int_t> &nodelist_h);

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

#ifdef USE_TILING   
   // Communication Work space 
   // In theory don't need CPU side buffer
  //  Real_t *commDataSend ;
  //  Real_t *commDataRecv ;

   Real_t *d_commDataSendMimic ;
   Real_t *d_commDataRecvMimic ;

   // Maximum number of block neighbors 
  //  MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
  //  MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
  cudaEvent_t sendEvent[26] ; // 6 faces + 12 edges + 8 corners //check if seding finished
#endif

};

typedef Real_t& (Domain::* Domain_member )(Index_t) ;

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

static inline void checkErrors(Domain* domain,int its,int myRank)
{
  if (*(domain->bad_vol_h) != -1)
  {
    printf("Rank %i: Volume Error in cell %d at iteration %d\n",myRank,*(domain->bad_vol_h),its);
    exit(VolumeError);
  }

  if (*(domain->bad_q_h) != -1)
  {
    printf("Rank %i: Q Error in cell %d at iteration %d\n",myRank,*(domain->bad_q_h),its);
    exit(QStopError);
  }
}
// cpu-comms
void MimicCommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void MimicCommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void MimicCommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void MimicCommSyncPosVel(Domain& domain);
void MimicCommMonoQ(Domain& domain);

// gpu-comms
void MimicCommSendGpu(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly, cudaStream_t stream);
void MimicCommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData, cudaStream_t *streams);
void MimicCommSyncPosVelGpu(Domain& domain, cudaStream_t *streams);
void MimicCommMonoQGpu(Domain& domain, cudaStream_t stream);

// Device helper functions for computation
__device__ inline real4 SQRT(real4 arg);
__device__ inline real8 SQRT(real8 arg);
__device__ inline real4 CBRT(real4 arg);
__device__ inline real8 CBRT(real8 arg);
__device__ __host__ inline real4 FABS(real4 arg) { return fabsf(arg); }
__device__ __host__ inline real8 FABS(real8 arg) { return fabs(arg); }
__device__ inline real4 FMAX(real4 arg1, real4 arg2);
__device__ inline real8 FMAX(real8 arg1, real8 arg2);

// Core computation functions
__host__ __device__ Real_t CalcElemVolume(const Real_t x[8], const Real_t y[8], const Real_t z[8]);
void CalcKinematicsForElems(Domain& domain, Real_t deltaTime, Index_t numElem);
void CalcLagrangeElements(Domain& domain, Real_t* vnew);
void CalcQForElems(Domain& domain);
void ApplyMaterialPropertiesForElems(Domain& domain);
void CalcTimeConstraintsForElems(Domain* domain);
void CalcAccelerationForNodes(Domain* domain);
void InitStressTermsForElems(Domain& domain, Real_t *sigxx, Real_t *sigyy, Real_t *sigzz);
void IntegrateStressForElems(Domain& domain, Real_t *sigxx, Real_t *sigyy, Real_t *sigzz, Real_t *determ);
void CalcHourglassControlForElems(Domain& domain, Real_t *hgcoef);
void CalcVolumeForceForElems(Domain& domain);
void CalcForceForNodes(Domain* domain);
void CalcMonotonicQRegionForElems(Domain& domain, Int_t r, Int_t rep);

// Main execution functions
void TimeIncrement(Domain* domain);
void LagrangeLeapFrog(Domain* domain);
void LagrangeNodal(Domain* domain);
void LagrangeElements(Domain* domain);

void ApplyMaterialPropertiesAndUpdateVolume(Domain *domain);
void CalcPositionAndVelocityForNodes(const Real_t u_cut, Domain* domain);
void CalcVolumeForceForElems(const Real_t hgcoef,Domain *domain);
// Initialization and setup functions
Domain *NewDomain(char* argv[], Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, bool structured, Int_t nr, Int_t balance, Int_t cost);
void InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t *col, Int_t *row, Int_t *plane, Int_t *side);
void printUsage(char *argv[]);
void cuda_init(Int_t device);
void CalcKinematicsAndMonotonicQGradient(Domain *domain);

#endif // LULESH_H
