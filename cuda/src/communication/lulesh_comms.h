#ifndef LULESH_COMMS_H
#define LULESH_COMMS_H

#include "../split/lulesh_split.h"

// Function declarations for communication-related files
// CPU-based communication functions

/**
 * @brief Receive domain boundary data from neighboring domains
 * @param domain The simulation domain
 * @param msgType Type of message being received (determines which boundary data)
 * @param xferFields Number of fields to transfer
 * @param dx X-dimension of the domain
 * @param dy Y-dimension of the domain
 * @param dz Z-dimension of the domain
 * @param doRecv Flag to control whether reception is performed
 * @param planeOnly Flag to control whether only plane data is exchanged (vs. edges and corners)
 */
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);

/**
 * @brief Send domain boundary data to neighboring domains
 * @param domain The simulation domain
 * @param msgType Type of message being sent (determines which boundary data)
 * @param xferFields Number of fields to transfer
 * @param fieldData Pointers to domain data fields being transferred
 * @param dx X-dimension of the domain
 * @param dy Y-dimension of the domain
 * @param dz Z-dimension of the domain
 * @param doSend Flag to control whether sending is performed
 * @param planeOnly Flag to control whether only plane data is exchanged (vs. edges and corners)
 */
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);

/**
 * @brief Perform a symmetric boundary nodal communication
 * @param domain The simulation domain
 * @param xferFields Number of fields to transfer
 * @param fieldData Pointers to domain data fields being transferred
 */
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);

/**
 * @brief Synchronize position and velocity data across domain boundaries
 * @param domain The simulation domain
 */
void CommSyncPosVel(Domain& domain);

/**
 * @brief Communicate data needed for monotonic q calculation across domain boundaries
 * @param domain The simulation domain
 */
void CommMonoQ(Domain& domain);

// GPU-specific communication functions

/**
 * @brief GPU version of CommSend that uses CUDA streams for asynchronous operation
 * @param domain The simulation domain
 * @param msgType Type of message being sent (determines which boundary data)
 * @param xferFields Number of fields to transfer
 * @param fieldData Pointers to domain data fields being transferred
 * @param dx X-dimension of the domain
 * @param dy Y-dimension of the domain
 * @param dz Z-dimension of the domain
 * @param doSend Flag to control whether sending is performed
 * @param planeOnly Flag to control whether only plane data is exchanged (vs. edges and corners)
 * @param stream CUDA stream to use for asynchronous operations
 */
void CommSendGpu(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly, cudaStream_t stream);

/**
 * @brief GPU version of CommSBN using CUDA streams for asynchronous operation
 * @param domain The simulation domain
 * @param xferFields Number of fields to transfer
 * @param fieldData Pointers to domain data fields being transferred
 * @param streams Array of CUDA streams for parallel execution
 */
void CommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData, cudaStream_t *streams);

/**
 * @brief GPU version of CommSyncPosVel using CUDA streams for asynchronous operation
 * @param domain The simulation domain
 * @param streams Array of CUDA streams for parallel execution
 */
void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams);

/**
 * @brief GPU version of CommMonoQ using a CUDA stream for asynchronous operation
 * @param domain The simulation domain
 * @param stream CUDA stream to use for asynchronous operations
 */
void CommMonoQGpu(Domain& domain, cudaStream_t stream);

#endif // LULESH_COMMS_H