#pragma once

#include "nperf/types.h"
#include <memory>
#include <string>

namespace nperf {

/// Topology detection engine
/// Discovers GPU topology using NVML, CUDA, and system tools
class TopologyDetector {
public:
    TopologyDetector();
    ~TopologyDetector();

    /// Detect full topology
    TopologyInfo detect();

    /// Detect GPUs only
    std::vector<GPUInfo> detectGPUs();

    /// Detect P2P matrix
    std::vector<std::vector<P2PInfo>> detectP2PMatrix();

    /// Detect NVLinks for a specific GPU
    std::vector<NVLinkInfo> detectNVLinks(int gpuIndex);

    /// Detect RDMA devices
    std::vector<RDMAInfo> detectRDMA();

    /// Check for NVSwitch presence
    bool hasNVSwitch();

    /// Get hostname
    std::string getHostname();

    /// Get NCCL version
    void getNcclVersion(int& major, int& minor, int& patch);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Parse nvidia-smi topo -m output (fallback)
std::vector<std::vector<P2PInfo>> parseNvidiaSmiTopo();

/// Determine link type from P2P performance rank
LinkType determineLinkType(int gpu1, int gpu2, int perfRank, bool hasNvlink);

} // namespace nperf
