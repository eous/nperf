#include "nperf/coordination/mpi_coordinator.h"

#ifdef NPERF_HAS_MPI

#include <stdexcept>
#include <cstring>

namespace nperf {

MPICoordinator::MPICoordinator() = default;

MPICoordinator::~MPICoordinator() {
    if (initialized_ && ownsMpi_) {
        finalize();
    }
}

void MPICoordinator::initialize(int argc, char** argv) {
    if (initialized_) {
        return;
    }

    // Check if MPI is already initialized
    int alreadyInitialized;
    MPI_Initialized(&alreadyInitialized);

    if (!alreadyInitialized) {
        // Initialize MPI with thread support
        int provided;
        int err = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (err != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed");
        }

        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            throw std::runtime_error("MPI does not provide required thread support");
        }

        ownsMpi_ = true;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

    // Get hostname
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(hostname, &len);
    hostname_ = std::string(hostname, len);

    initialized_ = true;
}

void MPICoordinator::finalize() {
    if (initialized_ && ownsMpi_) {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
        ownsMpi_ = false;
    }
    initialized_ = false;
}

void MPICoordinator::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void MPICoordinator::broadcastNcclId(ncclUniqueId* id, int root) {
    MPI_Bcast(id, sizeof(ncclUniqueId), MPI_BYTE, root, MPI_COMM_WORLD);
}

void MPICoordinator::broadcast(void* data, size_t size, int root) {
    MPI_Bcast(data, static_cast<int>(size), MPI_BYTE, root, MPI_COMM_WORLD);
}

void MPICoordinator::allReduceSum(double* data, size_t count) {
    MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(count),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void MPICoordinator::allReduceSum(int64_t* data, size_t count) {
    MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(count),
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

void MPICoordinator::gather(const void* send, void* recv, size_t size, int root) {
    MPI_Gather(send, static_cast<int>(size), MPI_BYTE,
               recv, static_cast<int>(size), MPI_BYTE,
               root, MPI_COMM_WORLD);
}

} // namespace nperf

#endif // NPERF_HAS_MPI
