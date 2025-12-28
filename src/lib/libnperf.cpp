/// @file libnperf.cpp
/// @brief Implementation of library interface (mostly header-only)

#include "nperf/nperf.h"

// This file exists to ensure the library has at least one compilation unit
// Most functionality is in headers or other source files

namespace nperf {

// Version info
extern "C" {

const char* nperf_version() {
    return NPERF_VERSION;
}

int nperf_version_major() {
    return NPERF_VERSION_MAJOR;
}

int nperf_version_minor() {
    return NPERF_VERSION_MINOR;
}

int nperf_version_patch() {
    return NPERF_VERSION_PATCH;
}

} // extern "C"

} // namespace nperf
