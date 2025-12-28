// NVLink-specific detection utilities
// Main detection is in detector.cpp; this file provides additional utilities

#include "nperf/topology/detector.h"
#include <nvml.h>

namespace nperf {

/// Get NVLink bandwidth in GB/s based on version and lane count
double getNvlinkBandwidth(int version, int lanes) {
    // Per-lane bandwidth varies by NVLink version
    double perLaneBandwidth;
    switch (version) {
        case 1: perLaneBandwidth = 20.0;  break;  // Pascal (P100)
        case 2: perLaneBandwidth = 25.0;  break;  // Volta (V100)
        case 3: perLaneBandwidth = 25.0;  break;  // Ampere (A100)
        case 4: perLaneBandwidth = 50.0;  break;  // Hopper (H100)
        case 5: perLaneBandwidth = 100.0; break;  // Blackwell (B100)
        default: perLaneBandwidth = 25.0; break;
    }
    return perLaneBandwidth * lanes;
}

/// Format NVLink info for display (e.g., "NV12" for 12 lanes)
std::string formatNvlinkType(int lanes, int version) {
    if (version > 0) {
        return "NV" + std::to_string(lanes) + "v" + std::to_string(version);
    }
    return "NV" + std::to_string(lanes);
}

} // namespace nperf
