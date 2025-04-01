#include "tiled_domain.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

/**
 * Main entry point for the tiled LULESH simulation
 * 
 * Command-line arguments:
 * -x INT: Number of tiles in X dimension (default: 2)
 * -y INT: Number of tiles in Y dimension (default: 2)
 * -z INT: Number of tiles in Z dimension (default: 2)
 * -n INT: Size of each tile in elements (default: 30)
 * -i INT: Maximum number of iterations (default: 1000)
 * -r INT: Number of regions per domain (default: 11)
 * -b INT: Region balance algorithm (default: 1)
 * -c INT: Cost multiplier (default: 1)
 * -h: Print help message
 */
int main(int argc, char* argv[])
{
    // Default parameters
    Int_t tilesX = 2;         // Number of tiles in X dimension
    Int_t tilesY = 2;         // Number of tiles in Y dimension
    Int_t tilesZ = 2;         // Number of tiles in Z dimension
    Int_t nx = 30;            // Size of each tile in elements (per dimension)
    Int_t maxIterations = 1000;// Maximum number of iterations
    Real_t stopTime = -1.0;   // Negative means run until max iterations
    Int_t numReg = 11;        // Number of regions per domain
    Int_t balance = 1;        // Region balance setting
    Int_t cost = 1;           // Cost multiplier
    Int_t maxStreams = 8;     // Maximum number of CUDA streams
    
    // Parse command-line arguments
    int opt;
    while ((opt = getopt(argc, argv, "x:y:z:n:i:t:r:b:c:s:h")) != -1) {
        switch (opt) {
            case 'x':
                tilesX = atoi(optarg);
                break;
            case 'y':
                tilesY = atoi(optarg);
                break;
            case 'z':
                tilesZ = atoi(optarg);
                break;
            case 'n':
                nx = atoi(optarg);
                break;
            case 'i':
                maxIterations = atoi(optarg);
                break;
            case 't':
                stopTime = atof(optarg);
                break;
            case 'r':
                numReg = atoi(optarg);
                break;
            case 'b':
                balance = atoi(optarg);
                break;
            case 'c':
                cost = atoi(optarg);
                break;
            case 's':
                maxStreams = atoi(optarg);
                break;
            case 'h':
                std::cout << "Tiled LULESH: CUDA implementation with tiled domains\n\n"
                          << "Options:\n"
                          << "  -x INT    Number of tiles in X dimension (default: 2)\n"
                          << "  -y INT    Number of tiles in Y dimension (default: 2)\n"
                          << "  -z INT    Number of tiles in Z dimension (default: 2)\n"
                          << "  -n INT    Size of each tile in elements (default: 30)\n"
                          << "  -i INT    Maximum number of iterations (default: 1000)\n"
                          << "  -t FLOAT  Stop time (default: run until max iterations)\n"
                          << "  -r INT    Number of regions per domain (default: 11)\n"
                          << "  -b INT    Region balance algorithm (default: 1)\n"
                          << "  -c INT    Cost multiplier (default: 1)\n"
                          << "  -s INT    Maximum number of CUDA streams (default: 8)\n"
                          << "  -h        Print this help message\n";
                return 0;
            default:
                std::cerr << "Unknown option: " << opt << std::endl;
                return 1;
        }
    }
    
    // Initialize CUDA
    cudaFree(0);  // Simple way to initialize CUDA
    
    // Create the tiled domain
    TiledDomain* tiledDomain = new TiledDomain(tilesX, tilesY, tilesZ, nx, numReg, balance, cost, maxStreams);
    
    // Print information about the tiled domain
    tiledDomain->PrintTiledDomainInfo();
    
    // Run the simulation
    tiledDomain->Run(stopTime, maxIterations);
    
    // Print summary information
    tiledDomain->PrintSummary();
    
    // Clean up
    delete tiledDomain;
    
    return 0;
}