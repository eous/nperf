#pragma once

#include "types.h"
#include "config.h"
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace nperf {

/// Timing result for a single iteration
struct IterationResult {
    double latencyUs = 0.0;       // Latency in microseconds
    size_t messageBytes = 0;
    bool verified = true;
    std::string verifyError;
};

/// Aggregated timing statistics
struct TimingStats {
    double avgUs = 0.0;
    double minUs = 0.0;
    double maxUs = 0.0;
    double stddevUs = 0.0;
    double p50Us = 0.0;           // Median
    double p95Us = 0.0;
    double p99Us = 0.0;
    int sampleCount = 0;

    /// Compute statistics from a list of latencies
    static TimingStats compute(const std::vector<double>& latencies) {
        TimingStats stats;
        if (latencies.empty()) return stats;

        stats.sampleCount = static_cast<int>(latencies.size());

        // Sort for percentiles
        std::vector<double> sorted = latencies;
        std::sort(sorted.begin(), sorted.end());

        stats.minUs = sorted.front();
        stats.maxUs = sorted.back();
        stats.avgUs = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();

        // Percentiles
        auto percentile = [&sorted](double p) {
            size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
            return sorted[idx];
        };
        stats.p50Us = percentile(0.50);
        stats.p95Us = percentile(0.95);
        stats.p99Us = percentile(0.99);

        // Standard deviation
        double variance = 0.0;
        for (double v : sorted) {
            variance += (v - stats.avgUs) * (v - stats.avgUs);
        }
        stats.stddevUs = std::sqrt(variance / sorted.size());

        return stats;
    }
};

/// Bandwidth metrics for a message size
struct BandwidthMetrics {
    double dataGBps = 0.0;        // Raw data bandwidth
    double algoGBps = 0.0;        // Algorithm bandwidth
    double busGBps = 0.0;         // Bus bandwidth (normalized)
};

/// Result for a single message size
struct SizeResult {
    size_t messageBytes = 0;
    size_t elementCount = 0;      // Number of elements (bytes / dtype size)
    int iterations = 0;

    TimingStats timing;
    BandwidthMetrics bandwidth;

    bool verified = true;
    int verifyErrors = 0;

    std::string detectedTransport;  // e.g., "NVLink", "P2P/IPC", "NET/IB"
};

/// Interval progress report
struct IntervalReport {
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    size_t bytesTransferred = 0;
    int operationsCompleted = 0;
    double currentBandwidthGBps = 0.0;
    double currentLatencyUs = 0.0;

    // Progress tracking
    size_t currentSizeIndex = 0;    // Which size is being tested (0-based)
    size_t totalSizes = 0;          // Total number of sizes
    size_t currentMessageBytes = 0; // Current message size being tested
    int currentIteration = 0;       // Current iteration within size
    int totalIterations = 0;        // Total iterations for this size
    double overallProgress = 0.0;   // 0.0 to 1.0 overall completion
};

/// Complete benchmark results
struct BenchmarkResults {
    // Configuration snapshot
    BenchmarkConfig config;
    TopologyInfo topology;

    // Per-size results
    std::vector<SizeResult> sizeResults;

    // Interval reports (for progress)
    std::vector<IntervalReport> intervals;

    // Summary statistics
    double peakBusGBps = 0.0;
    double avgBusGBps = 0.0;
    double totalBytes = 0.0;
    double totalTimeSeconds = 0.0;
    int totalIterations = 0;
    bool allVerified = true;
    int totalVerifyErrors = 0;

    // Timing
    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point endTime;

    // Rank info
    int rank = 0;
    int worldSize = 1;

    /// Compute summary statistics from size results
    void computeSummary() {
        if (sizeResults.empty()) return;

        peakBusGBps = 0.0;
        double sumBusGBps = 0.0;
        totalBytes = 0.0;
        totalIterations = 0;
        allVerified = true;
        totalVerifyErrors = 0;

        for (const auto& sr : sizeResults) {
            if (sr.bandwidth.busGBps > peakBusGBps) {
                peakBusGBps = sr.bandwidth.busGBps;
            }
            sumBusGBps += sr.bandwidth.busGBps;
            totalBytes += static_cast<double>(sr.messageBytes) * sr.iterations;
            totalIterations += sr.iterations;

            if (!sr.verified) allVerified = false;
            totalVerifyErrors += sr.verifyErrors;
        }

        avgBusGBps = sumBusGBps / sizeResults.size();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime);
        totalTimeSeconds = duration.count() / 1000.0;
    }
};

/// Compute bus bandwidth factor for collective operation
/// This normalizes bandwidth to account for the algorithm's data movement
inline double getBusBandwidthFactor(CollectiveOp op, int worldSize) {
    int n = worldSize;
    if (n <= 1) return 1.0;

    switch (op) {
        case CollectiveOp::AllReduce:
            // AllReduce: data moves 2*(n-1)/n times the input size
            return 2.0 * (n - 1) / n;

        case CollectiveOp::AllGather:
        case CollectiveOp::ReduceScatter:
            // Data moves (n-1)/n times the total data
            return static_cast<double>(n - 1) / n;

        case CollectiveOp::Broadcast:
        case CollectiveOp::Reduce:
            // Data moves once
            return 1.0;

        case CollectiveOp::AlltoAll:
            // All ranks send to all other ranks
            return static_cast<double>(n - 1) / n;

        case CollectiveOp::Gather:
        case CollectiveOp::Scatter:
            // Root receives/sends from/to all others
            return static_cast<double>(n - 1) / n;

        case CollectiveOp::SendRecv:
            // Point-to-point
            return 1.0;
    }
    return 1.0;
}

/// Compute bandwidth metrics from timing
inline BandwidthMetrics computeBandwidth(
    size_t bytes,
    double latencyUs,
    CollectiveOp op,
    int worldSize
) {
    BandwidthMetrics bw;

    if (latencyUs <= 0) return bw;

    double seconds = latencyUs / 1e6;
    double gbytes = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

    bw.dataGBps = gbytes / seconds;
    bw.algoGBps = bw.dataGBps;

    double factor = getBusBandwidthFactor(op, worldSize);
    bw.busGBps = bw.algoGBps * factor;

    return bw;
}

} // namespace nperf
