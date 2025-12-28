#include "nperf/core/metrics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>

namespace nperf {

void MetricsCalculator::addSample(size_t messageBytes, double latencyUs) {
    samples_[messageBytes].push_back(latencyUs);
}

SizeResult MetricsCalculator::computeResult(size_t messageBytes) const {
    SizeResult result;
    result.messageBytes = messageBytes;

    auto it = samples_.find(messageBytes);
    if (it == samples_.end() || it->second.empty()) {
        return result;
    }

    const auto& latencies = it->second;
    result.iterations = static_cast<int>(latencies.size());

    // Compute timing statistics
    result.timing = TimingStats::compute(latencies);

    // Compute bandwidth using average latency
    result.bandwidth = computeBandwidth(
        messageBytes,
        result.timing.avgUs,
        operation_,
        worldSize_
    );

    return result;
}

void MetricsCalculator::clear() {
    samples_.clear();
}

std::vector<double> MetricsCalculator::getLatencies(size_t messageBytes) const {
    auto it = samples_.find(messageBytes);
    if (it != samples_.end()) {
        return it->second;
    }
    return {};
}

} // namespace nperf
