#pragma once

/// @file log.h
/// @brief Simple logging utilities for nperf

#include <iostream>
#include <string>

namespace nperf {

/// Log levels for controlling output verbosity
enum class LogLevel {
    Silent = 0,  // No output
    Error = 1,   // Only errors
    Warning = 2, // Errors and warnings
    Info = 3,    // Errors, warnings, and info
    Debug = 4    // All messages including debug
};

/// Global log level (can be set by application)
inline LogLevel& globalLogLevel() {
    static LogLevel level = LogLevel::Warning;
    return level;
}

/// Set the global log level
inline void setLogLevel(LogLevel level) {
    globalLogLevel() = level;
}

/// Get the global log level
inline LogLevel getLogLevel() {
    return globalLogLevel();
}

/// Log an error message
inline void logError(const std::string& msg) {
    if (globalLogLevel() >= LogLevel::Error) {
        std::cerr << "[nperf ERROR] " << msg << std::endl;
    }
}

/// Log a warning message
inline void logWarning(const std::string& msg) {
    if (globalLogLevel() >= LogLevel::Warning) {
        std::cerr << "[nperf WARNING] " << msg << std::endl;
    }
}

/// Log an info message
inline void logInfo(const std::string& msg) {
    if (globalLogLevel() >= LogLevel::Info) {
        std::cerr << "[nperf INFO] " << msg << std::endl;
    }
}

/// Log a debug message
inline void logDebug(const std::string& msg) {
    if (globalLogLevel() >= LogLevel::Debug) {
        std::cerr << "[nperf DEBUG] " << msg << std::endl;
    }
}

} // namespace nperf
