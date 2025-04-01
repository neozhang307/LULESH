#ifndef __TILED_DOMAIN_H__
#define __TILED_DOMAIN_H__

#include "../split/lulesh_split.h"
#include <vector>

/**
 * TiledDomain - Class for managing multiple Domain instances on a single GPU
 * 
 * This class implements a tiled version of LULESH where multiple domains (tiles)
 * exist on a single GPU. Instead of using MPI for communication between domains,
 * tiles communicate directly through GPU memory.
 */
class TiledDomain {
public:
    // Constructor
    TiledDomain(Int_t tilesX, Int_t tilesY, Int_t tilesZ, 
                Int_t nx, Int_t numReg, Int_t balance, Int_t cost,
                Int_t maxStreams);
    
    // Destructor
    ~TiledDomain();
    
    // Run the simulation for specified time or iterations
    void Run(Real_t stopTime, Int_t maxIterations);
    
    // Access to domain tiles
    Domain* GetTile(Int_t z, Int_t y, Int_t x);
    
    // Debugging and output
    void PrintTiledDomainInfo();
    
    // Validation functions
    Real_t ComputeChecksum();
    void PrintSummary();

private:
    // 3D vector to store domains
    std::vector<std::vector<std::vector<Domain*>>> m_tiles;
    
    // Dimensions of the tile grid
    Int_t m_tilesX, m_tilesY, m_tilesZ;
    Int_t m_numTiles;  // Total number of tiles
    
    // Domain configuration parameters
    Int_t m_nx;        // Size of each domain
    Int_t m_numReg;    // Number of regions per domain
    Int_t m_balance;   // Region assignment algorithm
    Int_t m_cost;      // Cost multiplier
    Int_t m_maxStreams; // Maximum CUDA streams per domain
    
    // CUDA streams for concurrent execution
    std::vector<cudaStream_t> m_computeStreams; // Streams for computation
    std::vector<cudaStream_t> m_haloStreams;    // Streams for halo exchange
    std::vector<cudaEvent_t> m_computeEvents;   // Events for synchronization
    std::vector<cudaEvent_t> m_haloEvents;      // Events for synchronization
    
    // Initialize a single tile
    void InitializeTile(Int_t z, Int_t y, Int_t x);
    
    // Run a single simulation timestep across all tiles
    void RunStep();
    
    // Exchange boundaries between neighboring tiles
    void ExchangeTileBoundaries();
    
    // Calculate time constraints across all tiles
    void CalcTimeConstraints();
    
    // Helper functions for boundary exchange
    void CopyXFace(Domain* source, Domain* target, bool isPositive, cudaStream_t stream);
    void CopyYFace(Domain* source, Domain* target, bool isPositive, cudaStream_t stream);
    void CopyZFace(Domain* source, Domain* target, bool isPositive, cudaStream_t stream);
    
    // Corner and edge exchanges
    void CopyCorner(Domain* source, Domain* target, Int_t cornerType, cudaStream_t stream);
    void CopyEdge(Domain* source, Domain* target, Int_t edgeType, cudaStream_t stream);
    
    // GPU kernels for halo exchange are defined in tiled_domain.cu
};

#endif // __TILED_DOMAIN_H__