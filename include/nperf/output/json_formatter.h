#pragma once

#include "nperf/output/formatter.h"
#include <nlohmann/json.hpp>

namespace nperf {

/// JSON output formatter
class JsonFormatter : public Formatter {
public:
    explicit JsonFormatter(bool prettyPrint = false);

    std::string formatHeader(const NperfConfig& config,
                            const TopologyInfo& topology) override;

    std::string formatSizeResult(const SizeResult& result) override;

    std::string formatInterval(const IntervalReport& interval) override;

    std::string formatResults(const BenchmarkResults& results) override;

    std::string formatTopology(const TopologyInfo& topology) override;

private:
    bool prettyPrint_;

    nlohmann::json configToJson(const BenchmarkConfig& config);
    nlohmann::json topologyToJson(const TopologyInfo& topology);
    nlohmann::json sizeResultToJson(const SizeResult& result);
    nlohmann::json intervalToJson(const IntervalReport& interval);
    nlohmann::json resultsToJson(const BenchmarkResults& results);
};

} // namespace nperf
