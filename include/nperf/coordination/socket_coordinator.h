#pragma once

#include "nperf/coordination/coordinator.h"
#include <vector>
#include <string>

namespace nperf {

/// Socket-based coordinator for multi-node operation without MPI
class SocketCoordinator : public Coordinator {
public:
    SocketCoordinator();
    ~SocketCoordinator() override;

    /// Initialize as server (rank 0)
    void initializeServer(int port, int expectedClients);

    /// Initialize as client (rank > 0)
    void initializeClient(const std::string& serverHost, int port);

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

    /// Set configuration (must be called before initialize)
    void setServerMode(int port, int expectedClients);
    void setClientMode(const std::string& serverHost, int port);

private:
    bool isServer_ = false;
    int port_ = 5201;
    std::string serverHost_;
    int expectedClients_ = 0;

    int rank_ = 0;
    int worldSize_ = 1;
    std::string hostname_;

    int serverSocket_ = -1;
    int connectionSocket_ = -1;  // For client: connection to server
    std::vector<int> clientSockets_;  // For server: connections from clients

    bool initialized_ = false;

    void cleanup();

    // Helper methods
    bool sendAll(int socket, const void* data, size_t size);
    bool recvAll(int socket, void* data, size_t size);
};

} // namespace nperf
