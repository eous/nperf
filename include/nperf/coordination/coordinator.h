#pragma once

#include "nperf/types.h"
#include <nccl.h>
#include <memory>
#include <string>
#include <vector>

namespace nperf {

/// Abstract interface for multi-rank coordination
class Coordinator {
public:
    virtual ~Coordinator() = default;

    /// Initialize the coordinator
    virtual void initialize(int argc, char** argv) = 0;

    /// Finalize and cleanup
    virtual void finalize() = 0;

    /// Get this rank's ID
    virtual int getRank() const = 0;

    /// Get total number of ranks
    virtual int getWorldSize() const = 0;

    /// Get hostname for this rank
    virtual std::string getHostname() const = 0;

    /// Barrier synchronization
    virtual void barrier() = 0;

    /// Broadcast NCCL unique ID from root
    virtual void broadcastNcclId(ncclUniqueId* id, int root = 0) = 0;

    /// Broadcast data from root
    virtual void broadcast(void* data, size_t size, int root = 0) = 0;

    /// AllReduce for result aggregation (doubles)
    virtual void allReduceSum(double* data, size_t count) = 0;

    /// AllReduce for result aggregation (int64)
    virtual void allReduceSum(int64_t* data, size_t count) = 0;

    /// Gather data to root
    virtual void gather(const void* send, void* recv, size_t size, int root = 0) = 0;

    /// Check if coordinator is root rank
    bool isRoot() const { return getRank() == 0; }

    /// Factory method to create coordinator
    static std::unique_ptr<Coordinator> create(CoordinationMode mode);
};

/// Local coordinator for single-node operation
class LocalCoordinator : public Coordinator {
public:
    explicit LocalCoordinator(int numGpus = -1);

    void initialize(int argc, char** argv) override;
    void finalize() override;

    int getRank() const override { return 0; }
    int getWorldSize() const override { return 1; }
    std::string getHostname() const override;

    void barrier() override {}
    void broadcastNcclId(ncclUniqueId* id, int root = 0) override;
    void broadcast([[maybe_unused]] void* data, [[maybe_unused]] size_t size,
                   [[maybe_unused]] int root = 0) override {}
    void allReduceSum([[maybe_unused]] double* data, [[maybe_unused]] size_t count) override {}
    void allReduceSum([[maybe_unused]] int64_t* data, [[maybe_unused]] size_t count) override {}
    void gather(const void* send, void* recv, size_t size, int root = 0) override;

private:
    int numGpus_;
    std::string hostname_;
};

} // namespace nperf
