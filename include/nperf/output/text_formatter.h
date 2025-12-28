#pragma once

#include "nperf/output/formatter.h"

namespace nperf {

/// Human-readable text output formatter
class TextFormatter : public Formatter {
public:
    TextFormatter() = default;

    std::string formatHeader(const NperfConfig& config,
                            const TopologyInfo& topology) override;

    std::string formatSizeResult(const SizeResult& result) override;

    std::string formatInterval(const IntervalReport& interval) override;

    std::string formatResults(const BenchmarkResults& results) override;

    std::string formatTopology(const TopologyInfo& topology) override;

private:
    std::string formatP2PMatrix(const std::vector<std::vector<P2PInfo>>& matrix);
    std::string formatResultsTable(const std::vector<SizeResult>& results);
    std::string formatSummary(const BenchmarkResults& results);
};

} // namespace nperf
