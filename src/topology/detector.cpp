#include "nperf/topology/detector.h"
#include "nperf/log.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <nccl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace nperf {

class TopologyDetector::Impl {
public:
    Impl() {
        initNVML();
    }

    ~Impl() {
        shutdownNVML();
    }

    void initNVML() {
        nvmlReturn_t ret = nvmlInit();
        if (ret != NVML_SUCCESS) {
            logDebug("NVML initialization failed: " + std::string(nvmlErrorString(ret)));
            nvmlInitialized_ = false;
            return;
        }
        nvmlInitialized_ = true;

        unsigned int deviceCount;
        if (nvmlDeviceGetCount(&deviceCount) == NVML_SUCCESS) {
            nvmlDevices_.resize(deviceCount);
            for (unsigned int i = 0; i < deviceCount; i++) {
                nvmlDeviceGetHandleByIndex(i, &nvmlDevices_[i]);
            }
        }
    }

    void shutdownNVML() {
        if (nvmlInitialized_) {
            nvmlShutdown();
            nvmlInitialized_ = false;
        }
    }

    bool nvmlInitialized_ = false;
    std::vector<nvmlDevice_t> nvmlDevices_;
};

TopologyDetector::TopologyDetector()
    : impl_(std::make_unique<Impl>()) {
}

TopologyDetector::~TopologyDetector() = default;

TopologyInfo TopologyDetector::detect() {
    TopologyInfo info;

    logDebug("Getting hostname...");
    info.hostname = getHostname();
    logDebug("Hostname: " + info.hostname);

    logDebug("Getting NCCL version...");
    getNcclVersion(info.ncclVersionMajor, info.ncclVersionMinor, info.ncclVersionPatch);
    logInfo("NCCL version: " + std::to_string(info.ncclVersionMajor) + "." +
           std::to_string(info.ncclVersionMinor) + "." +
           std::to_string(info.ncclVersionPatch));

    logDebug("Detecting GPUs...");
    info.gpus = detectGPUs();
    logInfo("Detected " + std::to_string(info.gpus.size()) + " GPU(s)");

    logDebug("Building P2P connectivity matrix...");
    info.p2pMatrix = detectP2PMatrix();

    logDebug("Detecting RDMA devices...");
    info.rdmaDevices = detectRDMA();
    if (!info.rdmaDevices.empty()) {
        logInfo("Detected " + std::to_string(info.rdmaDevices.size()) + " RDMA device(s)");
    }

    logDebug("Checking for NVSwitch...");
    info.hasNVSwitch = hasNVSwitch();
    if (info.hasNVSwitch) {
        logInfo("NVSwitch detected");
    }

    info.discoveredAt = std::chrono::system_clock::now();

    return info;
}

std::vector<GPUInfo> TopologyDetector::detectGPUs() {
    std::vector<GPUInfo> gpus;

    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        return gpus;
    }

    for (int i = 0; i < deviceCount; i++) {
        GPUInfo gpu;
        gpu.deviceId = i;

        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess) {
            gpu.name = props.name;
            gpu.computeCapabilityMajor = props.major;
            gpu.computeCapabilityMinor = props.minor;
            gpu.totalMemoryBytes = props.totalGlobalMem;

            // Format PCI bus ID
            char pciBusId[20];
            snprintf(pciBusId, sizeof(pciBusId), "%04x:%02x:%02x.0",
                    props.pciDomainID, props.pciBusID, props.pciDeviceID);
            gpu.pciBusId = pciBusId;
        }

        // Get UUID from NVML
        if (impl_->nvmlInitialized_ && i < static_cast<int>(impl_->nvmlDevices_.size())) {
            char uuid[96];
            if (nvmlDeviceGetUUID(impl_->nvmlDevices_[i], uuid, sizeof(uuid)) == NVML_SUCCESS) {
                gpu.uuid = uuid;
            }
        }

        // Detect NVLinks
        gpu.nvlinks = detectNVLinks(i);
        gpu.nvlinkCount = static_cast<int>(gpu.nvlinks.size());

        // NUMA node (from NVML if available)
        if (impl_->nvmlInitialized_ && i < static_cast<int>(impl_->nvmlDevices_.size())) {
            nvmlPciInfo_t pciInfo;
            if (nvmlDeviceGetPciInfo(impl_->nvmlDevices_[i], &pciInfo) == NVML_SUCCESS) {
                // Try to get NUMA node from sysfs based on PCI bus ID
                // This is a simplified version; real implementation would read from sysfs
                gpu.numaNode = -1; // Unknown
            }
        }

        gpus.push_back(gpu);
    }

    return gpus;
}

std::vector<NVLinkInfo> TopologyDetector::detectNVLinks(int gpuIndex) {
    std::vector<NVLinkInfo> links;

    if (!impl_->nvmlInitialized_ ||
        gpuIndex >= static_cast<int>(impl_->nvmlDevices_.size())) {
        return links;
    }

    nvmlDevice_t device = impl_->nvmlDevices_[gpuIndex];

    // Check up to NVML_NVLINK_MAX_LINKS (typically 12-18)
    for (unsigned int link = 0; link < 18; link++) {
        nvmlEnableState_t state;
        nvmlReturn_t ret = nvmlDeviceGetNvLinkState(device, link, &state);

        if (ret != NVML_SUCCESS) {
            break; // No more links
        }

        if (state == NVML_FEATURE_ENABLED) {
            NVLinkInfo info;
            info.sourceGpu = gpuIndex;
            info.linkIndex = link;
            info.isActive = true;

            // Get NVLink version
            unsigned int version;
            if (nvmlDeviceGetNvLinkVersion(device, link, &version) == NVML_SUCCESS) {
                info.version = version;
            }

            // Get remote device info
            nvmlPciInfo_t remotePci;
            if (nvmlDeviceGetNvLinkRemotePciInfo(device, link, &remotePci) == NVML_SUCCESS) {
                info.remotePciBusId = remotePci.busId;

                // Try to find which GPU this connects to
                info.targetGpu = -1; // Unknown by default
                for (size_t j = 0; j < impl_->nvmlDevices_.size(); j++) {
                    nvmlPciInfo_t thisPci;
                    if (nvmlDeviceGetPciInfo(impl_->nvmlDevices_[j], &thisPci) == NVML_SUCCESS) {
                        if (strcmp(thisPci.busId, remotePci.busId) == 0) {
                            info.targetGpu = static_cast<int>(j);
                            break;
                        }
                    }
                }
            }

            links.push_back(info);
        }
    }

    return links;
}

std::vector<std::vector<P2PInfo>> TopologyDetector::detectP2PMatrix() {
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        return {};
    }

    std::vector<std::vector<P2PInfo>> matrix(deviceCount,
        std::vector<P2PInfo>(deviceCount));

    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            P2PInfo& info = matrix[i][j];
            info.gpu1 = i;
            info.gpu2 = j;

            if (i == j) {
                info.accessSupported = true;
                info.atomicSupported = true;
                info.performanceRank = 0;
                info.linkType = LinkType::Same;
                continue;
            }

            int accessSupported = 0;
            cudaDeviceGetP2PAttribute(&accessSupported,
                cudaDevP2PAttrAccessSupported, i, j);
            info.accessSupported = (accessSupported == 1);

            int atomicSupported = 0;
            cudaDeviceGetP2PAttribute(&atomicSupported,
                cudaDevP2PAttrNativeAtomicSupported, i, j);
            info.atomicSupported = (atomicSupported == 1);

            int perfRank = 0;
            cudaDeviceGetP2PAttribute(&perfRank,
                cudaDevP2PAttrPerformanceRank, i, j);
            info.performanceRank = perfRank;

            // Determine link type based on NVLink presence and perf rank
            bool hasNvlink = false;
            int nvlinkLanes = 0;
            int nvlinkVersion = 0;
            if (impl_->nvmlInitialized_ && i < static_cast<int>(impl_->nvmlDevices_.size())) {
                auto links = detectNVLinks(i);
                for (const auto& link : links) {
                    if (link.targetGpu == j) {
                        hasNvlink = true;
                        nvlinkLanes++;
                        nvlinkVersion = link.version;
                    }
                }
            }
            info.nvlinkVersion = nvlinkVersion;
            info.nvlinkLanes = nvlinkLanes;

            info.linkType = determineLinkType(i, j, perfRank, hasNvlink);
        }
    }

    return matrix;
}

std::vector<RDMAInfo> TopologyDetector::detectRDMA() {
    std::vector<RDMAInfo> devices;
    namespace fs = std::filesystem;

    // Try to detect InfiniBand devices using std::filesystem
    // Real implementation would use libibverbs if available
    const fs::path ibPath = "/sys/class/infiniband";

    try {
        if (!fs::exists(ibPath) || !fs::is_directory(ibPath)) {
            return devices;
        }

        for (const auto& entry : fs::directory_iterator(ibPath)) {
            if (!entry.is_directory()) {
                continue;
            }

            RDMAInfo info;
            info.deviceName = entry.path().filename().string();

            // Try to get port state
            fs::path statePath = entry.path() / "ports" / "1" / "state";
            if (fs::exists(statePath)) {
                std::ifstream stateFile(statePath);
                std::string state;
                if (std::getline(stateFile, state)) {
                    // Format: "4: ACTIVE" or similar
                    auto colonPos = state.find(':');
                    if (colonPos != std::string::npos && colonPos + 2 < state.size()) {
                        info.portState = state.substr(colonPos + 2);
                    }
                }
            }

            // Try to get rate
            fs::path ratePath = entry.path() / "ports" / "1" / "rate";
            if (fs::exists(ratePath)) {
                std::ifstream rateFile(ratePath);
                std::string rate;
                if (std::getline(rateFile, rate)) {
                    // Format: "200 Gb/sec"
                    try {
                        info.rateGbps = std::stod(rate);
                    } catch (const std::exception& e) {
                        logWarning("Failed to parse RDMA rate '" + rate + "': " + e.what());
                        info.rateGbps = 0.0;
                    }
                }
            }

            info.portNumber = 1;
            info.linkType = "IB";
            info.gdrSupported = true; // Assume GDR for mlx5

            devices.push_back(info);
        }
    } catch (const fs::filesystem_error& e) {
        logDebug("Failed to read InfiniBand sysfs: " + std::string(e.what()));
    }

    return devices;
}

bool TopologyDetector::hasNVSwitch() {
    if (!impl_->nvmlInitialized_) {
        return false;
    }

    // NVSwitch is indicated by NVLink connections that don't go to GPUs
    // Check if any GPU has NVLinks going to non-GPU devices
    for (size_t i = 0; i < impl_->nvmlDevices_.size(); i++) {
        auto links = detectNVLinks(static_cast<int>(i));
        for (const auto& link : links) {
            if (link.targetGpu == -1 && link.isActive) {
                // NVLink to something other than a GPU - likely NVSwitch
                return true;
            }
        }
    }

    return false;
}

std::string TopologyDetector::getHostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}

void TopologyDetector::getNcclVersion(int& major, int& minor, int& patch) {
    int version;
    ncclGetVersion(&version);
    major = version / 10000;
    minor = (version / 100) % 100;
    patch = version % 100;
}

LinkType determineLinkType(int gpu1, int gpu2, int perfRank, bool hasNvlink) {
    if (gpu1 == gpu2) {
        return LinkType::Same;
    }

    if (hasNvlink) {
        return LinkType::NVLink;
    }

    // Use performance rank as hint
    // Lower rank = better connection
    // Typical mapping:
    // 0-10: NVLink (already handled above)
    // 10-20: PIX (same PCIe switch)
    // 20-30: PXB (multiple PCIe switches)
    // 30-40: PHB (same CPU)
    // 40-50: NODE (cross NUMA)
    // 50+: SYS

    if (perfRank < 20) {
        return LinkType::PIX;
    } else if (perfRank < 30) {
        return LinkType::PXB;
    } else if (perfRank < 40) {
        return LinkType::PHB;
    } else if (perfRank < 50) {
        return LinkType::NODE;
    }
    return LinkType::SYS;
}

} // namespace nperf
