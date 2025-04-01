#ifndef LULESH_COMMS_H
#define LULESH_COMMS_H

#include "../split/lulesh_split.h"

// Function declarations for communication-related files
// CPU-based communication functions
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);

void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);

void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// GPU-specific communication functions
void CommSendGpu(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly, cudaStream_t stream);

void CommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData, cudaStream_t *streams);
void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams);
void CommMonoQGpu(Domain& domain, cudaStream_t stream);

#endif // LULESH_COMMS_H