#pragma once

#include "nperf/types.h"
#include <string>

namespace nperf {

/// Topology visualization utilities
class TopoVisualizer {
public:
    /// Generate DOT format for Graphviz
    static std::string toDot(const TopologyInfo& topology);

    /// Generate ASCII art representation
    static std::string toAscii(const TopologyInfo& topology);

    /// Generate matrix view (like nvidia-smi topo -m)
    static std::string toMatrix(const TopologyInfo& topology);

    /// Generate tree view (hierarchical)
    static std::string toTree(const TopologyInfo& topology);

    /// Format based on TopoFormat enum
    static std::string format(const TopologyInfo& topology, TopoFormat fmt);
};

} // namespace nperf
