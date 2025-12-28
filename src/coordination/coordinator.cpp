#include "nperf/coordination/coordinator.h"
#include <unistd.h>
#include <cstring>
#include <stdexcept>

// Forward declarations for specialized coordinators
#ifdef NPERF_HAS_MPI
#include "nperf/coordination/mpi_coordinator.h"
#endif
#include "nperf/coordination/socket_coordinator.h"

namespace nperf {

std::unique_ptr<Coordinator> Coordinator::create(CoordinationMode mode) {
    switch (mode) {
        case CoordinationMode::Local:
            return std::make_unique<LocalCoordinator>();

#ifdef NPERF_HAS_MPI
        case CoordinationMode::MPI:
            return std::make_unique<MPICoordinator>();
#endif

        case CoordinationMode::Socket:
            return std::make_unique<SocketCoordinator>();

        default:
            throw std::runtime_error("Unsupported coordination mode");
    }
}

// LocalCoordinator implementation

LocalCoordinator::LocalCoordinator(int numGpus) : numGpus_(numGpus) {
    if (numGpus_ < 0) {
        int count;
        cudaGetDeviceCount(&count);
        numGpus_ = count;
    }
}

void LocalCoordinator::initialize(int /*argc*/, char** /*argv*/) {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname_ = hostname;
    } else {
        hostname_ = "localhost";
    }
}

void LocalCoordinator::finalize() {
    // Nothing to do
}

std::string LocalCoordinator::getHostname() const {
    return hostname_;
}

void LocalCoordinator::broadcastNcclId(ncclUniqueId* id, int /*root*/) {
    // In local mode, just generate the ID
    ncclGetUniqueId(id);
}

void LocalCoordinator::gather(const void* send, void* recv, size_t size, int /*root*/) {
    // In local mode, just copy
    std::memcpy(recv, send, size);
}

} // namespace nperf
