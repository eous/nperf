#pragma once

#include "nperf/types.h"
#include "nperf/results.h"
#include <vector>
#include <map>

namespace nperf {

/// Metrics calculator for benchmark results
class MetricsCalculator {
public:
    /// Set the collective operation type (affects bus bandwidth calculation)
    void setOperation(CollectiveOp op) { operation_ = op; }

    /// Set the world size (number of ranks)
    void setWorldSize(int worldSize) { worldSize_ = worldSize; }

    /// Add a timing sample for a message size
    void addSample(size_t messageBytes, double latencyUs);

    /// Compute SizeResult for accumulated samples
    SizeResult computeResult(size_t messageBytes) const;

    /// Clear all samples
    void clear();

    /// Get all recorded latencies for a size
    std::vector<double> getLatencies(size_t messageBytes) const;

private:
    CollectiveOp operation_ = CollectiveOp::AllReduce;
    int worldSize_ = 1;

    // Map of message size -> vector of latency samples
    std::map<size_t, std::vector<double>> samples_;
};

/// Calculate effective element count based on operation
/// For some operations, count may differ from message bytes / element size
inline size_t effectiveCount(CollectiveOp op, size_t bytes, DataType dtype, int worldSize) {
    size_t elementSize = dataTypeSize(dtype);
    size_t totalElements = bytes / elementSize;

    switch (op) {
        case CollectiveOp::AllGather:
            // Each rank contributes bytes/worldSize
            return totalElements / worldSize;

        case CollectiveOp::ReduceScatter:
            // Each rank receives bytes/worldSize
            return totalElements / worldSize;

        case CollectiveOp::Gather:
        case CollectiveOp::Scatter:
            return totalElements / worldSize;

        default:
            return totalElements;
    }
}

/// Get operation-specific description of count
inline const char* countDescription(CollectiveOp op) {
    switch (op) {
        case CollectiveOp::AllGather:
            return "sendcount";
        case CollectiveOp::ReduceScatter:
            return "recvcount";
        case CollectiveOp::Gather:
            return "sendcount";
        case CollectiveOp::Scatter:
            return "recvcount";
        default:
            return "count";
    }
}

} // namespace nperf
