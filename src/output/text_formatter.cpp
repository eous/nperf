#include "nperf/output/text_formatter.h"
#include "nperf/version.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace nperf {

std::string TextFormatter::formatHeader(const NperfConfig& config,
                                        const TopologyInfo& topology) {
    std::ostringstream ss;

    // Banner
    ss << std::string(80, '=') << "\n";
    ss << std::setw(40 + 14) << std::right << NPERF_VERSION_STRING << " - NCCL Benchmark\n";
    ss << std::string(80, '=') << "\n\n";

    // Configuration
    ss << "Configuration:\n";
    ss << "  Operation:     " << collectiveOpName(config.benchmark.operation) << "\n";
    ss << "  Data Type:     " << dataTypeName(config.benchmark.dataType) << "\n";
    ss << "  Reduction:     " << reduceOpName(config.benchmark.reduceOp) << "\n";
    ss << "  Algorithm:     " << algorithmName(config.benchmark.algorithm) << "\n";
    ss << "  Protocol:      " << protocolName(config.benchmark.protocol) << "\n";
    ss << "  Size Range:    " << formatSize(config.benchmark.minBytes);
    if (config.benchmark.maxBytes != config.benchmark.minBytes) {
        ss << " - " << formatSize(config.benchmark.maxBytes);
    }
    ss << "\n";
    ss << "  Iterations:    " << config.benchmark.iterations << "\n";
    if (config.benchmark.useCudaGraph) {
        ss << "  CUDA Graph:    Enabled\n";
    }
    if (config.benchmark.verifyMode != VerifyMode::None) {
        ss << "  Verification:  Enabled\n";
    }
    ss << "\n";

    // Topology summary
    ss << "Topology:\n";
    ss << "  Host:          " << topology.hostname << "\n";
    ss << "  NCCL Version:  " << topology.ncclVersionMajor << "."
       << topology.ncclVersionMinor << "." << topology.ncclVersionPatch << "\n";
    ss << "  GPUs:          " << topology.gpus.size();
    if (!topology.gpus.empty()) {
        ss << "x " << topology.gpus[0].name;
    }
    ss << "\n";
    ss << "  NVSwitch:      " << (topology.hasNVSwitch ? "Yes" : "No") << "\n";
    if (!topology.rdmaDevices.empty()) {
        ss << "  RDMA:          " << topology.rdmaDevices.size() << "x "
           << topology.rdmaDevices[0].deviceName;
        if (topology.rdmaDevices[0].rateGbps > 0) {
            ss << " (" << topology.rdmaDevices[0].rateGbps << " Gb/s)";
        }
        ss << "\n";
    }
    ss << "\n";

    return ss.str();
}

std::string TextFormatter::formatSizeResult(const SizeResult& result) {
    std::ostringstream ss;

    ss << std::setw(10) << formatSize(result.messageBytes)
       << std::setw(12) << result.elementCount
       << std::fixed << std::setprecision(2)
       << std::setw(12) << result.timing.avgUs
       << std::setw(14) << result.bandwidth.algoGBps
       << std::setw(14) << result.bandwidth.busGBps
       << std::setw(10) << (result.verified ? "OK" : "FAIL")
       << "\n";

    return ss.str();
}

std::string TextFormatter::formatInterval(const IntervalReport& interval) {
    std::ostringstream ss;

    ss << "[" << std::fixed << std::setprecision(2)
       << interval.startSeconds << "-" << interval.endSeconds << "s] "
       << formatSize(interval.bytesTransferred) << " "
       << interval.currentBandwidthGBps << " GB/s\n";

    return ss.str();
}

std::string TextFormatter::formatResults(const BenchmarkResults& results) {
    std::ostringstream ss;

    // Header
    ss << formatHeader(
        NperfConfig{results.config, {}, {}},
        results.topology);

    // P2P Matrix (if we have it)
    if (!results.topology.p2pMatrix.empty()) {
        ss << "P2P Matrix:\n";
        ss << formatP2PMatrix(results.topology.p2pMatrix);
        ss << "\n";
    }

    // Results table
    ss << std::string(80, '-') << "\n";
    ss << std::setw(10) << "Size"
       << std::setw(12) << "Count"
       << std::setw(12) << "Time(us)"
       << std::setw(14) << "Algo BW(GB/s)"
       << std::setw(14) << "Bus BW(GB/s)"
       << std::setw(10) << "Status"
       << "\n";
    ss << std::string(80, '-') << "\n";

    for (const auto& sr : results.sizeResults) {
        ss << formatSizeResult(sr);
    }

    ss << std::string(80, '-') << "\n\n";

    // Summary
    ss << formatSummary(results);

    ss << std::string(80, '=') << "\n";

    return ss.str();
}

std::string TextFormatter::formatTopology(const TopologyInfo& topology) {
    std::ostringstream ss;

    ss << "GPU Topology for " << topology.hostname << "\n";
    ss << std::string(60, '-') << "\n\n";

    // GPU list
    ss << "GPUs:\n";
    for (const auto& gpu : topology.gpus) {
        ss << "  [" << gpu.deviceId << "] " << gpu.name << "\n";
        ss << "      PCI: " << gpu.pciBusId << "\n";
        ss << "      Memory: " << (gpu.totalMemoryBytes / (1024*1024*1024)) << " GB\n";
        ss << "      Compute: " << gpu.computeCapabilityMajor << "."
           << gpu.computeCapabilityMinor << "\n";
        if (gpu.nvlinkCount > 0) {
            ss << "      NVLinks: " << gpu.nvlinkCount << "\n";
        }
        if (gpu.numaNode >= 0) {
            ss << "      NUMA: " << gpu.numaNode << "\n";
        }
    }
    ss << "\n";

    // P2P Matrix
    if (!topology.p2pMatrix.empty()) {
        ss << "P2P Connectivity Matrix:\n";
        ss << formatP2PMatrix(topology.p2pMatrix);
        ss << "\n";
    }

    // Legend
    ss << "Legend:\n";
    ss << "  X    = Self\n";
    ss << "  NVx  = NVLink with x lanes\n";
    ss << "  NVS  = NVSwitch\n";
    ss << "  PIX  = Same PCI switch\n";
    ss << "  PXB  = Multiple PCI switches\n";
    ss << "  PHB  = Same CPU/host bridge\n";
    ss << "  NODE = Cross NUMA node\n";
    ss << "  SYS  = System interconnect\n";
    ss << "\n";

    // RDMA devices
    if (!topology.rdmaDevices.empty()) {
        ss << "RDMA Devices:\n";
        for (const auto& rdma : topology.rdmaDevices) {
            ss << "  " << rdma.deviceName << ": " << rdma.linkType;
            if (rdma.rateGbps > 0) {
                ss << " " << rdma.rateGbps << " Gb/s";
            }
            ss << " [" << rdma.portState << "]";
            if (!rdma.affinityGpus.empty()) {
                ss << " (GPU affinity:";
                for (int g : rdma.affinityGpus) {
                    ss << " " << g;
                }
                ss << ")";
            }
            ss << "\n";
        }
    }

    return ss.str();
}

std::string TextFormatter::formatP2PMatrix(
    const std::vector<std::vector<P2PInfo>>& matrix
) {
    if (matrix.empty()) return "";

    std::ostringstream ss;
    size_t n = matrix.size();

    // Header row
    ss << "        ";
    for (size_t i = 0; i < n; i++) {
        ss << std::setw(6) << ("GPU" + std::to_string(i));
    }
    ss << "\n";

    // Data rows
    for (size_t i = 0; i < n; i++) {
        ss << "  GPU" << i << " ";
        for (size_t j = 0; j < n; j++) {
            const auto& p2p = matrix[i][j];
            std::string label;

            if (i == j) {
                label = "X";
            } else if (p2p.linkType == LinkType::NVLink) {
                label = "NV" + std::to_string(p2p.nvlinkLanes);
            } else {
                label = linkTypeLegend(p2p.linkType);
            }

            ss << std::setw(6) << label;
        }
        ss << "\n";
    }

    return ss.str();
}

std::string TextFormatter::formatSummary(const BenchmarkResults& results) {
    std::ostringstream ss;

    ss << "Summary:\n";
    ss << "  Peak Bus Bandwidth:    " << std::fixed << std::setprecision(2)
       << results.peakBusGBps << " GB/s\n";
    ss << "  Average Bandwidth:     " << results.avgBusGBps << " GB/s\n";
    ss << "  Total Data:            " << formatSize(static_cast<size_t>(results.totalBytes)) << "\n";
    ss << "  Total Time:            " << results.totalTimeSeconds << " s\n";
    ss << "  Total Iterations:      " << results.totalIterations << "\n";

    if (results.config.verifyMode != VerifyMode::None) {
        ss << "  Verification:          "
           << (results.allVerified ? "PASSED" : "FAILED")
           << " (" << results.totalVerifyErrors << " errors)\n";
    }

    ss << "\n";

    return ss.str();
}

} // namespace nperf
