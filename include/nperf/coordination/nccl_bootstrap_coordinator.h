#pragma once

#include "coordinator.h"
#include <string>
#include <vector>

namespace nperf {

/// Coordinator using NCCL's native bootstrap mechanism
/// Uses NCCL_COMM_ID environment variable for bootstrap address
class NcclBootstrapCoordinator : public Coordinator {
public:
    NcclBootstrapCoordinator();
    ~NcclBootstrapCoordinator() override;

    /// Set rank and world size (must be called before initialize)
    void setRankInfo(int rank, int worldSize);

    void initialize(int argc, char** argv) override;
    void finalize() override;

    int getRank() const override { return rank_; }
    int getWorldSize() const override { return worldSize_; }
    std::string getHostname() const override { return hostname_; }

    void barrier() override;
    void broadcastNcclId(ncclUniqueId* id, int root = 0) override;
    void broadcast(void* data, size_t size, int root = 0) override;
    void allReduceSum(double* data, size_t count) override;
    void allReduceSum(int64_t* data, size_t count) override;
    void gather(const void* send, void* recv, size_t size, int root = 0) override;

private:
    int rank_ = -1;
    int worldSize_ = -1;
    std::string hostname_;
    bool initialized_ = false;

    // Socket for bootstrap communication (all ranks connect to rank 0)
    int serverSocket_ = -1;
    int connectionSocket_ = -1;
    std::vector<int> clientSockets_;

    // Parse NCCL_COMM_ID environment variable
    bool parseCommId(std::string& host, int& port);

    // Socket helpers
    bool sendAll(int socket, const void* data, size_t size);
    bool recvAll(int socket, void* data, size_t size);
};

} // namespace nperf
