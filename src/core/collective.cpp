#include "nperf/core/collective.h"
#include <stdexcept>
#include <string>

namespace nperf {

void checkNccl(ncclResult_t result, const char* op) {
    if (result != ncclSuccess) {
        std::string msg = std::string(op) + " failed: " + ncclGetErrorString(result);
        throw std::runtime_error(msg);
    }
}

// CollectiveRunner implementation

CollectiveRunner::CollectiveRunner(ncclComm_t comm, cudaStream_t stream)
    : comm_(comm), stream_(stream) {

    NCCL_CHECK(ncclCommUserRank(comm_, &rank_));
    NCCL_CHECK(ncclCommCount(comm_, &worldSize_));
}

ncclResult_t CollectiveRunner::run(
    CollectiveOp op,
    const void* sendBuf,
    void* recvBuf,
    size_t count,
    DataType dtype,
    ReduceOp redOp,
    int root
) {
    switch (op) {
        case CollectiveOp::AllReduce:
            return allReduce(sendBuf, recvBuf, count, dtype, redOp);

        case CollectiveOp::AllGather:
            return allGather(sendBuf, recvBuf, count, dtype);

        case CollectiveOp::Broadcast:
            // For broadcast, sendBuf and recvBuf are the same
            return broadcast(recvBuf, count, dtype, root);

        case CollectiveOp::Reduce:
            return reduce(sendBuf, recvBuf, count, dtype, redOp, root);

        case CollectiveOp::ReduceScatter:
            return reduceScatter(sendBuf, recvBuf, count, dtype, redOp);

        case CollectiveOp::AlltoAll:
            return allToAll(sendBuf, recvBuf, count, dtype);

        case CollectiveOp::Gather:
            return gather(sendBuf, recvBuf, count, dtype, root);

        case CollectiveOp::Scatter:
            return scatter(sendBuf, recvBuf, count, dtype, root);

        case CollectiveOp::SendRecv:
            // For send/recv, we do a ring pattern by default
            // Send to (rank+1) % worldSize, recv from (rank-1+worldSize) % worldSize
            {
                int sendPeer = (rank_ + 1) % worldSize_;
                // Note: recvPeer is implicitly calculated inside sendRecv based on NCCL behavior
                return sendRecv(sendBuf, count, sendPeer, recvBuf, count, dtype);
            }
    }
    return ncclSuccess;
}

ncclResult_t CollectiveRunner::allReduce(
    const void* sendBuf, void* recvBuf,
    size_t count, DataType dtype, ReduceOp op
) {
    return ncclAllReduce(sendBuf, recvBuf, count,
                         toNcclDataType(dtype), toNcclRedOp(op),
                         comm_, stream_);
}

ncclResult_t CollectiveRunner::allGather(
    const void* sendBuf, void* recvBuf,
    size_t sendCount, DataType dtype
) {
    return ncclAllGather(sendBuf, recvBuf, sendCount,
                         toNcclDataType(dtype), comm_, stream_);
}

ncclResult_t CollectiveRunner::broadcast(
    void* buf, size_t count,
    DataType dtype, int root
) {
    return ncclBroadcast(buf, buf, count,
                         toNcclDataType(dtype), root, comm_, stream_);
}

ncclResult_t CollectiveRunner::reduce(
    const void* sendBuf, void* recvBuf,
    size_t count, DataType dtype, ReduceOp op, int root
) {
    return ncclReduce(sendBuf, recvBuf, count,
                      toNcclDataType(dtype), toNcclRedOp(op),
                      root, comm_, stream_);
}

ncclResult_t CollectiveRunner::reduceScatter(
    const void* sendBuf, void* recvBuf,
    size_t recvCount, DataType dtype, ReduceOp op
) {
    return ncclReduceScatter(sendBuf, recvBuf, recvCount,
                             toNcclDataType(dtype), toNcclRedOp(op),
                             comm_, stream_);
}

ncclResult_t CollectiveRunner::allToAll(
    const void* sendBuf, void* recvBuf,
    size_t count, DataType dtype
) {
    // ncclAllToAll is available in NCCL 2.11+
    // For older versions, we would need to implement using send/recv
    #if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
    return ncclAlltoAll(sendBuf, recvBuf, count,
                        toNcclDataType(dtype), comm_, stream_);
    #else
    // Fallback implementation using grouped send/recv
    ncclGroupStart();
    size_t elementSize = dataTypeSize(dtype);
    for (int i = 0; i < worldSize_; i++) {
        const char* sendPtr = static_cast<const char*>(sendBuf) + i * count * elementSize;
        char* recvPtr = static_cast<char*>(recvBuf) + i * count * elementSize;
        ncclSend(sendPtr, count, toNcclDataType(dtype), i, comm_, stream_);
        ncclRecv(recvPtr, count, toNcclDataType(dtype), i, comm_, stream_);
    }
    return ncclGroupEnd();
    #endif
}

ncclResult_t CollectiveRunner::gather(
    const void* sendBuf, void* recvBuf,
    size_t sendCount, DataType dtype, int root
) {
    // Gather is not a native NCCL op, implement with send/recv
    ncclResult_t result = ncclSuccess;
    ncclGroupStart();

    if (rank_ == root) {
        size_t elementSize = dataTypeSize(dtype);
        for (int i = 0; i < worldSize_; i++) {
            char* ptr = static_cast<char*>(recvBuf) + i * sendCount * elementSize;
            if (i == root) {
                // Copy local data
                cudaError_t err = cudaMemcpyAsync(ptr, sendBuf, sendCount * elementSize,
                                                 cudaMemcpyDeviceToDevice, stream_);
                if (err != cudaSuccess) {
                    ncclGroupEnd();  // End group before returning error
                    return ncclInternalError;
                }
            } else {
                ncclRecv(ptr, sendCount, toNcclDataType(dtype), i, comm_, stream_);
            }
        }
    } else {
        ncclSend(sendBuf, sendCount, toNcclDataType(dtype), root, comm_, stream_);
    }

    result = ncclGroupEnd();
    return result;
}

ncclResult_t CollectiveRunner::scatter(
    const void* sendBuf, void* recvBuf,
    size_t recvCount, DataType dtype, int root
) {
    // Scatter is not a native NCCL op, implement with send/recv
    ncclResult_t result = ncclSuccess;
    ncclGroupStart();

    if (rank_ == root) {
        size_t elementSize = dataTypeSize(dtype);
        for (int i = 0; i < worldSize_; i++) {
            const char* ptr = static_cast<const char*>(sendBuf) + i * recvCount * elementSize;
            if (i == root) {
                cudaError_t err = cudaMemcpyAsync(recvBuf, ptr, recvCount * elementSize,
                                                 cudaMemcpyDeviceToDevice, stream_);
                if (err != cudaSuccess) {
                    ncclGroupEnd();  // End group before returning error
                    return ncclInternalError;
                }
            } else {
                ncclSend(ptr, recvCount, toNcclDataType(dtype), i, comm_, stream_);
            }
        }
    } else {
        ncclRecv(recvBuf, recvCount, toNcclDataType(dtype), root, comm_, stream_);
    }

    result = ncclGroupEnd();
    return result;
}

ncclResult_t CollectiveRunner::sendRecv(
    const void* sendBuf, size_t sendCount, int peer,
    void* recvBuf, size_t recvCount, DataType dtype
) {
    int recvPeer = (rank_ - 1 + worldSize_) % worldSize_;
    if (peer == rank_) {
        // Self send/recv - just copy
        size_t bytes = sendCount * dataTypeSize(dtype);
        cudaError_t err = cudaMemcpyAsync(recvBuf, sendBuf, bytes, cudaMemcpyDeviceToDevice, stream_);
        if (err != cudaSuccess) {
            return ncclInternalError;
        }
        return ncclSuccess;
    }

    ncclGroupStart();
    ncclSend(sendBuf, sendCount, toNcclDataType(dtype), peer, comm_, stream_);
    ncclRecv(recvBuf, recvCount, toNcclDataType(dtype), recvPeer, comm_, stream_);
    return ncclGroupEnd();
}

int CollectiveRunner::rank() const {
    return rank_;
}

int CollectiveRunner::worldSize() const {
    return worldSize_;
}

void CollectiveRunner::synchronize() {
    cudaStreamSynchronize(stream_);
}

// NcclCommunicator implementation

void NcclCommunicator::initAll(int numDevices, const int* deviceList) {
    destroy();

    // If deviceList is null, use devices 0..numDevices-1
    std::vector<int> defaultDevices;
    if (!deviceList) {
        defaultDevices.resize(numDevices);
        for (int i = 0; i < numDevices; i++) {
            defaultDevices[i] = i;
        }
        deviceList = defaultDevices.data();
    }

    // Note: ncclCommInitAll creates multiple communicators
    // For simplicity, we just create one for the current device
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    // Get current device
    int device;
    cudaGetDevice(&device);

    // Find rank
    int rank = -1;
    for (int i = 0; i < numDevices; i++) {
        if (deviceList[i] == device) {
            rank = i;
            break;
        }
    }

    if (rank < 0) {
        throw std::runtime_error("Current device not in device list");
    }

    NCCL_CHECK(ncclCommInitRank(&comm_, numDevices, id, rank));
}

void NcclCommunicator::initRank(int numRanks, ncclUniqueId commId, int rank) {
    destroy();
    NCCL_CHECK(ncclCommInitRank(&comm_, numRanks, commId, rank));
}

NcclCommunicator::~NcclCommunicator() {
    destroy();
}

NcclCommunicator::NcclCommunicator(NcclCommunicator&& other) noexcept
    : comm_(other.comm_) {
    other.comm_ = nullptr;
}

NcclCommunicator& NcclCommunicator::operator=(NcclCommunicator&& other) noexcept {
    if (this != &other) {
        destroy();
        comm_ = other.comm_;
        other.comm_ = nullptr;
    }
    return *this;
}

int NcclCommunicator::rank() const {
    if (!comm_) return -1;
    int r;
    ncclCommUserRank(comm_, &r);
    return r;
}

int NcclCommunicator::worldSize() const {
    if (!comm_) return 0;
    int s;
    ncclCommCount(comm_, &s);
    return s;
}

void NcclCommunicator::abort() {
    if (comm_) {
        ncclCommAbort(comm_);
        comm_ = nullptr;
    }
}

void NcclCommunicator::destroy() {
    if (comm_) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
}

ncclUniqueId NcclCommunicator::getUniqueId() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return id;
}

} // namespace nperf
