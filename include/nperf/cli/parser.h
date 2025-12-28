#pragma once

#include "nperf/config.h"
#include <string>
#include <vector>

namespace nperf {

/// Parse command line arguments into NperfConfig
class ArgParser {
public:
    ArgParser();

    /// Parse command line arguments
    /// Returns true if parsing succeeded, false if should exit (--help, error)
    bool parse(int argc, char** argv);

    /// Get parsed configuration
    const NperfConfig& config() const { return config_; }

    /// Check if help was requested
    bool helpRequested() const { return helpRequested_; }

    /// Check if version was requested
    bool versionRequested() const { return versionRequested_; }

    /// Get error message if parsing failed
    const std::string& errorMessage() const { return errorMessage_; }

private:
    NperfConfig config_;
    bool helpRequested_ = false;
    bool versionRequested_ = false;
    std::string errorMessage_;
};

/// Print usage information
void printUsage();

/// Print version information
void printVersion();

} // namespace nperf
