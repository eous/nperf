#include "nperf/output/topo_visualizer.h"
#include <sstream>
#include <iomanip>
#include <set>
#include <map>
#include <algorithm>

namespace nperf {

std::string TopoVisualizer::format(const TopologyInfo& topology, TopoFormat fmt) {
    switch (fmt) {
        case TopoFormat::Matrix:
            return toMatrix(topology);
        case TopoFormat::Tree:
            return toTree(topology);
        case TopoFormat::DOT:
            return toDot(topology);
        case TopoFormat::JSON:
            // JSON is handled by JsonFormatter
            return "";
    }
    return toMatrix(topology);
}

std::string TopoVisualizer::toDot(const TopologyInfo& topology) {
    std::ostringstream ss;

    ss << "digraph GPUTopology {\n";
    ss << "  rankdir=LR;\n";
    ss << "  node [shape=box, style=filled];\n";
    ss << "  edge [dir=none];\n\n";

    // GPU nodes
    ss << "  // GPU nodes\n";
    for (const auto& gpu : topology.gpus) {
        ss << "  GPU" << gpu.deviceId
           << " [label=\"" << gpu.name << "\\n"
           << "CUDA:" << gpu.deviceId << "\\n"
           << "PCI:" << gpu.pciBusId << "\", "
           << "fillcolor=lightblue];\n";
    }
    ss << "\n";

    // NVSwitch if present
    if (topology.hasNVSwitch) {
        ss << "  // NVSwitch\n";
        ss << "  NVSwitch [label=\"NVSwitch\", shape=diamond, fillcolor=lightgreen];\n";
        for (const auto& gpu : topology.gpus) {
            ss << "  GPU" << gpu.deviceId << " -> NVSwitch "
               << "[style=dashed, color=green];\n";
        }
        ss << "\n";
    }

    // NVLink connections (deduplicated)
    ss << "  // NVLink connections\n";
    std::set<std::pair<int, int>> nvlinkPairs;
    for (const auto& gpu : topology.gpus) {
        for (const auto& link : gpu.nvlinks) {
            if (link.targetGpu >= 0) {
                int src = std::min(gpu.deviceId, link.targetGpu);
                int dst = std::max(gpu.deviceId, link.targetGpu);
                nvlinkPairs.insert({src, dst});
            }
        }
    }

    for (const auto& pair : nvlinkPairs) {
        // Count lanes between this pair
        int lanes = 0;
        int version = 0;
        for (const auto& gpu : topology.gpus) {
            if (gpu.deviceId == pair.first) {
                for (const auto& link : gpu.nvlinks) {
                    if (link.targetGpu == pair.second) {
                        lanes++;
                        version = link.version;
                    }
                }
            }
        }

        ss << "  GPU" << pair.first << " -> GPU" << pair.second
           << " [label=\"NV" << lanes;
        if (version > 0) {
            ss << "v" << version;
        }
        ss << "\", color=green, penwidth=2];\n";
    }
    ss << "\n";

    // RDMA devices
    if (!topology.rdmaDevices.empty()) {
        ss << "  // RDMA devices\n";
        for (size_t i = 0; i < topology.rdmaDevices.size(); i++) {
            const auto& rdma = topology.rdmaDevices[i];
            ss << "  RDMA" << i << " [label=\"" << rdma.deviceName << "\\n"
               << rdma.linkType;
            if (rdma.rateGbps > 0) {
                ss << " " << rdma.rateGbps << "Gb/s";
            }
            ss << "\", shape=ellipse, fillcolor=lightyellow];\n";

            // GPU-NIC affinity
            for (int gpuId : rdma.affinityGpus) {
                ss << "  GPU" << gpuId << " -> RDMA" << i
                   << " [style=dotted, color=orange];\n";
            }
        }
    }

    ss << "}\n";

    return ss.str();
}

std::string TopoVisualizer::toAscii(const TopologyInfo& topology) {
    // Simple ASCII representation
    return toTree(topology);
}

std::string TopoVisualizer::toMatrix(const TopologyInfo& topology) {
    std::ostringstream ss;

    size_t n = topology.gpus.size();
    if (n == 0) return "No GPUs detected\n";

    // Column width
    const int colWidth = 6;

    // Header
    ss << std::setw(8) << "";
    for (size_t i = 0; i < n; i++) {
        ss << std::setw(colWidth) << ("GPU" + std::to_string(i));
    }

    // Add NIC columns if present
    for (size_t i = 0; i < topology.rdmaDevices.size(); i++) {
        ss << std::setw(colWidth) << ("NIC" + std::to_string(i));
    }

    ss << std::setw(16) << "CPU Affinity";
    ss << std::setw(8) << "NUMA";
    ss << "\n";

    // Rows
    for (size_t i = 0; i < n; i++) {
        ss << std::setw(6) << ("GPU" + std::to_string(i)) << "  ";

        for (size_t j = 0; j < n; j++) {
            std::string label;
            if (i == j) {
                label = "X";
            } else if (!topology.p2pMatrix.empty()) {
                const auto& p2p = topology.p2pMatrix[i][j];
                if (p2p.linkType == LinkType::NVLink) {
                    label = "NV" + std::to_string(p2p.nvlinkLanes);
                } else {
                    label = linkTypeLegend(p2p.linkType);
                }
            } else {
                label = "?";
            }
            ss << std::setw(colWidth) << label;
        }

        // NIC columns
        for (size_t k = 0; k < topology.rdmaDevices.size(); k++) {
            const auto& nic = topology.rdmaDevices[k];
            bool hasAffinity = std::find(nic.affinityGpus.begin(),
                                        nic.affinityGpus.end(),
                                        static_cast<int>(i)) != nic.affinityGpus.end();
            ss << std::setw(colWidth) << (hasAffinity ? "PIX" : "SYS");
        }

        // CPU affinity and NUMA
        const auto& gpu = topology.gpus[i];
        ss << std::setw(16) << "-"; // CPU affinity (would need more info)
        ss << std::setw(8) << (gpu.numaNode >= 0 ? std::to_string(gpu.numaNode) : "-");
        ss << "\n";
    }

    return ss.str();
}

std::string TopoVisualizer::toTree(const TopologyInfo& topology) {
    std::ostringstream ss;

    ss << "Host: " << topology.hostname << "\n";

    // Group GPUs by NUMA node
    std::map<int, std::vector<int>> numaToGpus;
    for (const auto& gpu : topology.gpus) {
        numaToGpus[gpu.numaNode].push_back(gpu.deviceId);
    }

    for (const auto& [numa, gpuIds] : numaToGpus) {
        if (numa >= 0) {
            ss << "├── NUMA Node " << numa << "\n";
        } else {
            ss << "├── NUMA Node (unknown)\n";
        }

        for (size_t i = 0; i < gpuIds.size(); i++) {
            int gpuId = gpuIds[i];
            const auto& gpu = topology.gpus[gpuId];
            bool isLast = (i == gpuIds.size() - 1);

            ss << "│   " << (isLast ? "└" : "├") << "── GPU" << gpuId
               << ": " << gpu.name << " (" << gpu.pciBusId << ")\n";

            // Show NVLinks
            if (!gpu.nvlinks.empty()) {
                for (size_t j = 0; j < gpu.nvlinks.size(); j++) {
                    const auto& link = gpu.nvlinks[j];
                    ss << "│   " << (isLast ? " " : "│")
                       << "   " << (j == gpu.nvlinks.size() - 1 ? "└" : "├")
                       << "── NVLink";
                    if (link.version > 0) {
                        ss << " v" << link.version;
                    }
                    ss << " -> ";
                    if (link.targetGpu >= 0) {
                        ss << "GPU" << link.targetGpu;
                    } else {
                        ss << "NVSwitch";
                    }
                    ss << "\n";
                }
            }
        }
    }

    // Network interfaces
    if (!topology.rdmaDevices.empty()) {
        ss << "└── Network Interfaces\n";
        for (size_t i = 0; i < topology.rdmaDevices.size(); i++) {
            const auto& nic = topology.rdmaDevices[i];
            bool isLast = (i == topology.rdmaDevices.size() - 1);
            ss << "    " << (isLast ? "└" : "├") << "── " << nic.deviceName
               << ": " << nic.linkType;
            if (nic.rateGbps > 0) {
                ss << " " << nic.rateGbps << " Gb/s";
            }
            if (!nic.portState.empty()) {
                ss << " [" << nic.portState << "]";
            }
            ss << "\n";
        }
    }

    return ss.str();
}

} // namespace nperf
