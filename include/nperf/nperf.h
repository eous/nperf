#pragma once

/// @file nperf.h
/// @brief Public API for the nperf NCCL benchmarking library

#include "nperf/version.h"
#include "nperf/types.h"
#include "nperf/config.h"
#include "nperf/results.h"
#include "nperf/core/engine.h"
#include "nperf/topology/detector.h"
#include "nperf/output/formatter.h"
#include "nperf/output/topo_visualizer.h"

/// @namespace nperf
/// @brief NCCL performance benchmarking utilities
namespace nperf {

/// @brief Simple API to run a benchmark
/// @param config Benchmark configuration
/// @param argc Command line argument count (for MPI initialization)
/// @param argv Command line arguments
/// @return Benchmark results
inline BenchmarkResults runBenchmark(const NperfConfig& config, int argc = 0, char** argv = nullptr) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.initialize(argc, argv);
    auto results = engine.run();
    engine.finalize();
    return results;
}

/// @brief Simple API to detect topology
/// @return Detected topology information
inline TopologyInfo detectTopology() {
    TopologyDetector detector;
    return detector.detect();
}

/// @brief Get NCCL version as string
/// @return Version string in "major.minor.patch" format
inline std::string getNcclVersionString() {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();
    return std::to_string(info.ncclVersionMajor) + "." +
           std::to_string(info.ncclVersionMinor) + "." +
           std::to_string(info.ncclVersionPatch);
}

/// @brief Get nperf library version
/// @return Version string
inline const char* getVersion() {
    return NPERF_VERSION;
}

} // namespace nperf
