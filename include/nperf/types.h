#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

namespace nperf {

// Forward declarations
struct GPUInfo;
struct NVLinkInfo;
struct P2PInfo;
struct RDMAInfo;
struct TopologyInfo;

/// NCCL collective operation types
enum class CollectiveOp {
    AllReduce,
    AllGather,
    Broadcast,
    Reduce,
    ReduceScatter,
    AlltoAll,
    Gather,
    Scatter,
    SendRecv
};

/// Get string name for collective operation
inline const char* collectiveOpName(CollectiveOp op) {
    switch (op) {
        case CollectiveOp::AllReduce:     return "allreduce";
        case CollectiveOp::AllGather:     return "allgather";
        case CollectiveOp::Broadcast:     return "broadcast";
        case CollectiveOp::Reduce:        return "reduce";
        case CollectiveOp::ReduceScatter: return "reducescatter";
        case CollectiveOp::AlltoAll:      return "alltoall";
        case CollectiveOp::Gather:        return "gather";
        case CollectiveOp::Scatter:       return "scatter";
        case CollectiveOp::SendRecv:      return "sendrecv";
    }
    return "unknown";
}

/// Parse collective operation from string
inline CollectiveOp parseCollectiveOp(const std::string& name) {
    if (name == "allreduce" || name == "all-reduce" || name == "all_reduce")
        return CollectiveOp::AllReduce;
    if (name == "allgather" || name == "all-gather" || name == "all_gather")
        return CollectiveOp::AllGather;
    if (name == "broadcast")
        return CollectiveOp::Broadcast;
    if (name == "reduce")
        return CollectiveOp::Reduce;
    if (name == "reducescatter" || name == "reduce-scatter" || name == "reduce_scatter")
        return CollectiveOp::ReduceScatter;
    if (name == "alltoall" || name == "all-to-all" || name == "all_to_all")
        return CollectiveOp::AlltoAll;
    if (name == "gather")
        return CollectiveOp::Gather;
    if (name == "scatter")
        return CollectiveOp::Scatter;
    if (name == "sendrecv" || name == "send-recv" || name == "send_recv")
        return CollectiveOp::SendRecv;
    return CollectiveOp::AllReduce; // default
}

/// NCCL data types
enum class DataType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int32,
    UInt32,
    Int64,
    UInt64
};

/// Get string name for data type
inline const char* dataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:  return "float32";
        case DataType::Float64:  return "float64";
        case DataType::Float16:  return "float16";
        case DataType::BFloat16: return "bfloat16";
        case DataType::Int8:     return "int8";
        case DataType::UInt8:    return "uint8";
        case DataType::Int32:    return "int32";
        case DataType::UInt32:   return "uint32";
        case DataType::Int64:    return "int64";
        case DataType::UInt64:   return "uint64";
    }
    return "unknown";
}

/// Get size in bytes for data type
inline size_t dataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:  return 4;
        case DataType::Float64:  return 8;
        case DataType::Float16:  return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int8:     return 1;
        case DataType::UInt8:    return 1;
        case DataType::Int32:    return 4;
        case DataType::UInt32:   return 4;
        case DataType::Int64:    return 8;
        case DataType::UInt64:   return 8;
    }
    return 4;
}

/// Parse data type from string
inline DataType parseDataType(const std::string& name) {
    if (name == "float32" || name == "float" || name == "f32")
        return DataType::Float32;
    if (name == "float64" || name == "double" || name == "f64")
        return DataType::Float64;
    if (name == "float16" || name == "half" || name == "f16")
        return DataType::Float16;
    if (name == "bfloat16" || name == "bf16")
        return DataType::BFloat16;
    if (name == "int8" || name == "i8")
        return DataType::Int8;
    if (name == "uint8" || name == "u8")
        return DataType::UInt8;
    if (name == "int32" || name == "int" || name == "i32")
        return DataType::Int32;
    if (name == "uint32" || name == "u32")
        return DataType::UInt32;
    if (name == "int64" || name == "i64")
        return DataType::Int64;
    if (name == "uint64" || name == "u64")
        return DataType::UInt64;
    return DataType::Float32;
}

/// NCCL reduction operations
enum class ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    Avg
};

/// Get string name for reduce operation
inline const char* reduceOpName(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:  return "sum";
        case ReduceOp::Prod: return "prod";
        case ReduceOp::Min:  return "min";
        case ReduceOp::Max:  return "max";
        case ReduceOp::Avg:  return "avg";
    }
    return "unknown";
}

/// Parse reduce operation from string
inline ReduceOp parseReduceOp(const std::string& name) {
    if (name == "sum")  return ReduceOp::Sum;
    if (name == "prod") return ReduceOp::Prod;
    if (name == "min")  return ReduceOp::Min;
    if (name == "max")  return ReduceOp::Max;
    if (name == "avg")  return ReduceOp::Avg;
    return ReduceOp::Sum;
}

/// NCCL algorithm
enum class Algorithm {
    Auto,
    Ring,
    Tree,
    CollNetDirect,
    CollNetChain,
    NVLS
};

inline const char* algorithmName(Algorithm algo) {
    switch (algo) {
        case Algorithm::Auto:          return "auto";
        case Algorithm::Ring:          return "ring";
        case Algorithm::Tree:          return "tree";
        case Algorithm::CollNetDirect: return "collnetdirect";
        case Algorithm::CollNetChain:  return "collnetchain";
        case Algorithm::NVLS:          return "nvls";
    }
    return "auto";
}

/// NCCL protocol
enum class Protocol {
    Auto,
    Simple,
    LL,      // Low Latency
    LL128    // Low Latency 128
};

inline const char* protocolName(Protocol proto) {
    switch (proto) {
        case Protocol::Auto:   return "auto";
        case Protocol::Simple: return "simple";
        case Protocol::LL:     return "ll";
        case Protocol::LL128:  return "ll128";
    }
    return "auto";
}

/// Coordination mode
enum class CoordinationMode {
    Local,         // Single-node, no coordination needed
    MPI,           // MPI-based coordination
    Socket,        // TCP socket-based coordination
    NcclBootstrap  // NCCL native bootstrap via NCCL_COMM_ID
};

/// Link type between GPUs (matches nvidia-smi topo legend)
enum class LinkType {
    Same,       // X - Same device
    NVLink,     // NVx - NVLink with x lanes
    NVSwitch,   // NVS - NVSwitch fabric
    C2C,        // C2C - Chip-to-chip (Grace Hopper)
    PIX,        // PIX - Same PCI switch
    PXB,        // PXB - Multiple PCI switches (no host bridge)
    PHB,        // PHB - Same CPU/host bridge
    NODE,       // NODE - Cross NUMA node
    SYS,        // SYS - System interconnect
    NET         // NET - Network (InfiniBand, etc.)
};

/// Get legend string for link type (like nvidia-smi)
inline const char* linkTypeLegend(LinkType type) {
    switch (type) {
        case LinkType::Same:     return "X";
        case LinkType::NVLink:   return "NV";
        case LinkType::NVSwitch: return "NVS";
        case LinkType::C2C:      return "C2C";
        case LinkType::PIX:      return "PIX";
        case LinkType::PXB:      return "PXB";
        case LinkType::PHB:      return "PHB";
        case LinkType::NODE:     return "NODE";
        case LinkType::SYS:      return "SYS";
        case LinkType::NET:      return "NET";
    }
    return "?";
}

/// NVLink information for a single link
struct NVLinkInfo {
    int sourceGpu = -1;
    int targetGpu = -1;    // -1 if connected to NVSwitch or CPU
    int linkIndex = 0;
    int version = 0;       // NVLink version (1, 2, 3, 4)
    bool isActive = false;
    std::string remotePciBusId;
};

/// P2P (peer-to-peer) connectivity info between two GPUs
struct P2PInfo {
    int gpu1 = 0;
    int gpu2 = 0;
    bool accessSupported = false;
    bool atomicSupported = false;
    int performanceRank = 0;    // Lower is better, 0 = best
    LinkType linkType = LinkType::SYS;
    int nvlinkLanes = 0;        // If linkType == NVLink
    int nvlinkVersion = 0;
};

/// RDMA/InfiniBand device information
struct RDMAInfo {
    std::string deviceName;     // e.g., "mlx5_0"
    std::string portState;      // "Active", "Down", etc.
    int portNumber = 1;
    std::string linkType;       // "IB", "Ethernet", "RoCE"
    double rateGbps = 0.0;      // e.g., 200.0 for HDR
    uint64_t guid = 0;
    std::vector<int> affinityGpus;  // GPUs with NUMA affinity to this NIC
    bool gdrSupported = false;      // GPUDirect RDMA support
};

/// GPU device information
struct GPUInfo {
    int deviceId = 0;
    std::string name;           // e.g., "NVIDIA A100-SXM4-80GB"
    std::string uuid;           // GPU-xxx-xxx format
    std::string pciBusId;       // e.g., "0000:3b:00.0"
    int computeCapabilityMajor = 0;
    int computeCapabilityMinor = 0;
    size_t totalMemoryBytes = 0;
    int numaNode = -1;
    int nvlinkCount = 0;
    bool gdrSupported = false;
    std::vector<NVLinkInfo> nvlinks;
};

/// Complete topology information
struct TopologyInfo {
    std::string hostname;
    int ncclVersionMajor = 0;
    int ncclVersionMinor = 0;
    int ncclVersionPatch = 0;
    bool hasNVSwitch = false;
    std::vector<GPUInfo> gpus;
    std::vector<std::vector<P2PInfo>> p2pMatrix;  // [gpu1][gpu2]
    std::vector<RDMAInfo> rdmaDevices;
    std::chrono::system_clock::time_point discoveredAt;
};

/// Output format
enum class OutputFormat {
    Text,       // Human-readable table
    JSON,       // Structured JSON
    JSONPretty  // Pretty-printed JSON
};

/// Topology visualization format
enum class TopoFormat {
    Matrix,     // Like nvidia-smi topo -m
    Tree,       // Hierarchical tree view
    DOT,        // Graphviz DOT format
    JSON        // JSON export
};

/// Verification mode
enum class VerifyMode {
    None,           // No verification
    PostBenchmark,  // Verify after all iterations
    PerIteration    // Verify after each iteration
};

} // namespace nperf
