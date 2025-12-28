#pragma once

#include "nperf/types.h"
#include <nccl.h>
#include <cuda_runtime.h>

namespace nperf {

/// Convert nperf DataType to NCCL data type
inline ncclDataType_t toNcclDataType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:  return ncclFloat32;
        case DataType::Float64:  return ncclFloat64;
        case DataType::Float16:  return ncclFloat16;
        case DataType::BFloat16: return ncclBfloat16;
        case DataType::Int8:     return ncclInt8;
        case DataType::UInt8:    return ncclUint8;
        case DataType::Int32:    return ncclInt32;
        case DataType::UInt32:   return ncclUint32;
        case DataType::Int64:    return ncclInt64;
        case DataType::UInt64:   return ncclUint64;
    }
    return ncclFloat32;
}

/// Convert nperf ReduceOp to NCCL reduce operation
inline ncclRedOp_t toNcclRedOp(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:  return ncclSum;
        case ReduceOp::Prod: return ncclProd;
        case ReduceOp::Min:  return ncclMin;
        case ReduceOp::Max:  return ncclMax;
        case ReduceOp::Avg:  return ncclAvg;
    }
    return ncclSum;
}

/// Check NCCL result and throw on error
void checkNccl(ncclResult_t result, const char* op);

/// Macro for NCCL error checking
#define NCCL_CHECK(cmd) nperf::checkNccl((cmd), #cmd)

/// Wrapper for NCCL collective operations
class CollectiveRunner {
public:
    /// Create runner with existing communicator and stream
    CollectiveRunner(ncclComm_t comm, cudaStream_t stream);

    /// Run a collective operation
    ncclResult_t run(
        CollectiveOp op,
        const void* sendBuf,
        void* recvBuf,
        size_t count,
        DataType dtype,
        ReduceOp redOp = ReduceOp::Sum,
        int root = 0
    );

    // Individual collective operations
    ncclResult_t allReduce(const void* sendBuf, void* recvBuf,
                           size_t count, DataType dtype, ReduceOp op);

    ncclResult_t allGather(const void* sendBuf, void* recvBuf,
                           size_t sendCount, DataType dtype);

    ncclResult_t broadcast(void* buf, size_t count,
                           DataType dtype, int root);

    ncclResult_t reduce(const void* sendBuf, void* recvBuf,
                        size_t count, DataType dtype, ReduceOp op, int root);

    ncclResult_t reduceScatter(const void* sendBuf, void* recvBuf,
                               size_t recvCount, DataType dtype, ReduceOp op);

    ncclResult_t allToAll(const void* sendBuf, void* recvBuf,
                          size_t count, DataType dtype);

    ncclResult_t gather(const void* sendBuf, void* recvBuf,
                        size_t sendCount, DataType dtype, int root);

    ncclResult_t scatter(const void* sendBuf, void* recvBuf,
                         size_t recvCount, DataType dtype, int root);

    ncclResult_t sendRecv(const void* sendBuf, size_t sendCount,
                          int peer, void* recvBuf, size_t recvCount,
                          DataType dtype);

    /// Get communicator info
    int rank() const;
    int worldSize() const;

    /// Get stream
    cudaStream_t stream() const { return stream_; }

    /// Synchronize stream
    void synchronize();

private:
    ncclComm_t comm_;
    cudaStream_t stream_;
    int rank_ = -1;
    int worldSize_ = -1;
};

/// RAII wrapper for NCCL communicator
class NcclCommunicator {
public:
    NcclCommunicator() = default;

    /// Initialize for single-node multi-GPU
    void initAll(int numDevices, const int* deviceList = nullptr);

    /// Initialize for multi-node (with external ID distribution)
    void initRank(int numRanks, ncclUniqueId commId, int rank);

    /// Destructor
    ~NcclCommunicator();

    // Non-copyable
    NcclCommunicator(const NcclCommunicator&) = delete;
    NcclCommunicator& operator=(const NcclCommunicator&) = delete;

    // Movable
    NcclCommunicator(NcclCommunicator&& other) noexcept;
    NcclCommunicator& operator=(NcclCommunicator&& other) noexcept;

    /// Get raw handle
    ncclComm_t handle() const { return comm_; }

    /// Check if initialized
    bool valid() const { return comm_ != nullptr; }

    /// Get rank
    int rank() const;

    /// Get world size
    int worldSize() const;

    /// Abort communicator
    void abort();

    /// Destroy communicator
    void destroy();

    /// Generate unique ID (for rank 0)
    static ncclUniqueId getUniqueId();

private:
    ncclComm_t comm_ = nullptr;
};

} // namespace nperf
