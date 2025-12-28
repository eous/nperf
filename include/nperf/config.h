#pragma once

#include "types.h"
#include "log.h"
#include <string>
#include <vector>
#include <cstdint>
#include <climits>

namespace nperf {

/// Configuration for benchmark run
struct BenchmarkConfig {
    // Collective operation settings
    CollectiveOp operation = CollectiveOp::AllReduce;
    DataType dataType = DataType::Float32;
    ReduceOp reduceOp = ReduceOp::Sum;
    Algorithm algorithm = Algorithm::Auto;
    Protocol protocol = Protocol::Auto;

    // Message size settings
    size_t minBytes = 1024;           // 1 KB default
    size_t maxBytes = 1024;           // Same as min for single size
    size_t stepFactor = 2;            // Multiply by 2 for each step

    // Duration settings (mutually exclusive)
    bool useTimeBased = false;
    double testDurationSeconds = 0.0; // Time-based mode
    int iterations = 20;              // Iteration-based mode

    // Warmup settings
    int warmupIterations = 5;
    double omitSeconds = 0.0;         // Omit from start (like iperf -O)

    // Progress reporting
    double reportIntervalSeconds = 1.0;

    // Verification
    VerifyMode verifyMode = VerifyMode::None;
    double verifyTolerance = 1e-5;

    // CUDA options
    bool useCudaGraph = false;        // CUDA Graph capture mode
    int cudaDevice = -1;              // -1 = auto select

    // Root rank for rooted operations (broadcast, reduce)
    int rootRank = 0;

    /// Generate list of message sizes to test
    std::vector<size_t> getMessageSizes() const {
        std::vector<size_t> sizes;
        if (minBytes == 0 || stepFactor <= 1) {
            // Single size or invalid step
            sizes.push_back(minBytes > 0 ? minBytes : 1);
            if (maxBytes != minBytes) {
                sizes.push_back(maxBytes);
            }
            return sizes;
        }

        for (size_t size = minBytes; size <= maxBytes; ) {
            sizes.push_back(size);
            // Overflow protection
            if (size > SIZE_MAX / stepFactor) break;
            size_t next = size * stepFactor;
            if (next <= size) break;  // Overflow happened
            size = next;
        }
        // Ensure maxBytes is included if not exact multiple
        if (sizes.empty() || sizes.back() != maxBytes) {
            sizes.push_back(maxBytes);
        }
        return sizes;
    }
};

/// Configuration for coordination
struct CoordinationConfig {
    CoordinationMode mode = CoordinationMode::Local;

    // Local mode settings
    int numLocalGpus = -1;  // -1 = all available

    // Socket mode settings
    std::string serverHost;
    int port = 5201;
    bool isServer = false;
    int expectedClients = 1;

    // MPI is initialized via command line arguments
};

/// Output configuration
struct OutputConfig {
    OutputFormat format = OutputFormat::Text;
    std::string outputFile;         // Empty = stdout
    bool showTopology = false;      // Show topology before benchmark
    TopoFormat topoFormat = TopoFormat::Matrix;
    bool topologyOnly = false;      // Just show topology, no benchmark
    bool showTransport = false;     // Show transport info during benchmark
    bool verbose = false;
    bool debug = false;             // Enable NCCL_DEBUG=INFO
};

/// Complete nperf configuration
struct NperfConfig {
    BenchmarkConfig benchmark;
    CoordinationConfig coordination;
    OutputConfig output;

    /// Validate configuration
    bool validate(std::string& error) const {
        if (benchmark.minBytes > benchmark.maxBytes) {
            error = "minBytes cannot be greater than maxBytes";
            return false;
        }
        if (benchmark.minBytes == 0) {
            error = "minBytes must be > 0";
            return false;
        }
        if (benchmark.iterations <= 0 && !benchmark.useTimeBased) {
            error = "iterations must be > 0";
            return false;
        }
        if (benchmark.useTimeBased && benchmark.testDurationSeconds <= 0) {
            error = "testDurationSeconds must be > 0 for time-based mode";
            return false;
        }
        if (coordination.mode == CoordinationMode::Socket &&
            !coordination.isServer && coordination.serverHost.empty()) {
            error = "serverHost required for socket client mode";
            return false;
        }
        return true;
    }
};

/// Parse size string with K/M/G suffixes
inline size_t parseSize(const std::string& str) {
    if (str.empty()) return 0;

    char suffix = str.back();
    size_t multiplier = 1;
    std::string numPart = str;

    if (suffix == 'K' || suffix == 'k') {
        multiplier = 1024;
        numPart = str.substr(0, str.size() - 1);
    } else if (suffix == 'M' || suffix == 'm') {
        multiplier = 1024 * 1024;
        numPart = str.substr(0, str.size() - 1);
    } else if (suffix == 'G' || suffix == 'g') {
        multiplier = 1024ULL * 1024 * 1024;
        numPart = str.substr(0, str.size() - 1);
    } else if (suffix == 'T' || suffix == 't') {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
        numPart = str.substr(0, str.size() - 1);
    }

    try {
        unsigned long long value = std::stoull(numPart);
        // Check for overflow before multiplication
        if (multiplier > 1 && value > SIZE_MAX / multiplier) {
            logWarning("Size value would overflow: " + str);
            return 0;
        }
        return static_cast<size_t>(value * multiplier);
    } catch (const std::exception& e) {
        logWarning("Failed to parse size '" + str + "': " + e.what());
        return 0;
    }
}

/// Format size with appropriate suffix
inline std::string formatSize(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && idx < 4) {
        size /= 1024.0;
        idx++;
    }

    char buf[32];
    if (size == static_cast<double>(static_cast<int>(size))) {
        snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(size), suffixes[idx]);
    } else {
        snprintf(buf, sizeof(buf), "%.2f %s", size, suffixes[idx]);
    }
    return std::string(buf);
}

} // namespace nperf
