#pragma once

#include "nperf/coordination/coordinator.h"

#ifdef NPERF_HAS_MPI
#include <mpi.h>
#endif

namespace nperf {

#ifdef NPERF_HAS_MPI

/// MPI-based coordinator for multi-node operation
class MPICoordinator : public Coordinator {
public:
    MPICoordinator();
    ~MPICoordinator() override;

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

    /// Check if MPI was initialized by us
    bool ownsMpi() const { return ownsMpi_; }

private:
    int rank_ = 0;
    int worldSize_ = 1;
    std::string hostname_;
    bool initialized_ = false;
    bool ownsMpi_ = false;
};

#else

// Stub when MPI is not available
class MPICoordinator : public Coordinator {
public:
    MPICoordinator() {
        throw std::runtime_error("MPI support not compiled in");
    }

    void initialize(int, char**) override {}
    void finalize() override {}
    int getRank() const override { return 0; }
    int getWorldSize() const override { return 1; }
    std::string getHostname() const override { return ""; }
    void barrier() override {}
    void broadcastNcclId(ncclUniqueId*, int) override {}
    void broadcast(void*, size_t, int) override {}
    void allReduceSum(double*, size_t) override {}
    void allReduceSum(int64_t*, size_t) override {}
    void gather(const void*, void*, size_t, int) override {}
};

#endif

} // namespace nperf
