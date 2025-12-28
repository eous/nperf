#pragma once

#include "nperf/config.h"
#include "nperf/results.h"
#include "nperf/types.h"
#include "nperf/coordination/coordinator.h"
#include "nperf/topology/detector.h"
#include "nperf/core/collective.h"
#include "nperf/core/memory.h"
#include "nperf/core/timing.h"
#include "nperf/core/graph.h"
#include "nperf/core/metrics.h"
#include "nperf/verification/verifier.h"
#include "nperf/output/formatter.h"
#include <memory>
#include <functional>

namespace nperf {

/// Callback for progress reporting
using ProgressCallback = std::function<void(const IntervalReport&)>;

/// Main benchmark orchestration engine
class BenchmarkEngine {
public:
    BenchmarkEngine();
    ~BenchmarkEngine();

    /// Set configuration
    void configure(const NperfConfig& config);

    /// Initialize engine (coordinator, NCCL, memory)
    void initialize(int argc, char** argv);

    /// Run the benchmark
    BenchmarkResults run();

    /// Run topology-only mode
    TopologyInfo runTopologyOnly();

    /// Set progress callback
    void setProgressCallback(ProgressCallback callback);

    /// Get detected topology
    const TopologyInfo& topology() const { return topology_; }

    /// Get rank info
    int rank() const;
    int worldSize() const;

    /// Finalize and cleanup
    void finalize();

private:
    NperfConfig config_;
    TopologyInfo topology_;

    std::unique_ptr<Coordinator> coordinator_;
    NcclCommunicator ncclComm_;
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<MemoryManager> memoryManager_;
    DeviceBuffer sendBuffer_;
    DeviceBuffer recvBuffer_;

    std::unique_ptr<CollectiveRunner> runner_;
    std::unique_ptr<GraphRunner> graphRunner_;
    std::unique_ptr<Verifier> verifier_;
    MetricsCalculator metrics_;

    ProgressCallback progressCallback_;

    bool initialized_ = false;

    // Internal methods
    void setupNccl();
    void allocateBuffers(size_t maxBytes);
    void captureGraph(size_t count);

    SizeResult runSize(size_t bytes);
    void runWarmup(size_t bytes);

    void setNcclEnvVars();
};

} // namespace nperf
