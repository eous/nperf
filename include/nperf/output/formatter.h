#pragma once

#include "nperf/types.h"
#include "nperf/config.h"
#include "nperf/results.h"
#include <string>
#include <ostream>
#include <memory>

namespace nperf {

/// Abstract base class for output formatters
class Formatter {
public:
    virtual ~Formatter() = default;

    /// Format header (configuration, topology info)
    virtual std::string formatHeader(const NperfConfig& config,
                                     const TopologyInfo& topology) = 0;

    /// Format a single size result
    virtual std::string formatSizeResult(const SizeResult& result) = 0;

    /// Format interval progress report
    virtual std::string formatInterval(const IntervalReport& interval) = 0;

    /// Format complete results with summary
    virtual std::string formatResults(const BenchmarkResults& results) = 0;

    /// Format topology information
    virtual std::string formatTopology(const TopologyInfo& topology) = 0;

    /// Factory method
    static std::unique_ptr<Formatter> create(OutputFormat format);
};

/// Get formatter for output format
std::unique_ptr<Formatter> createFormatter(OutputFormat format);

} // namespace nperf
