// RDMA/InfiniBand detection utilities

#include "nperf/topology/detector.h"
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <cstring>

namespace nperf {

/// Parse ibstat-like output to get HCA info
std::vector<RDMAInfo> parseIbstat() {
    std::vector<RDMAInfo> devices;

    // Check if ibstat is available
    FILE* fp = popen("ibstat 2>/dev/null", "r");
    if (!fp) {
        return devices;
    }

    RDMAInfo current;
    bool inDevice = false;
    bool inPort = false;
    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        std::string s = line;

        // Remove leading/trailing whitespace
        size_t start = s.find_first_not_of(" \t");
        size_t end = s.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        s = s.substr(start, end - start + 1);

        // CA 'mlx5_0'
        if (s.find("CA '") == 0) {
            if (inDevice && !current.deviceName.empty()) {
                devices.push_back(current);
            }
            current = RDMAInfo();
            size_t start = s.find("'") + 1;
            size_t end = s.rfind("'");
            current.deviceName = s.substr(start, end - start);
            inDevice = true;
            inPort = false;
            continue;
        }

        // Port 1:
        if (s.find("Port ") == 0 && s.back() == ':') {
            current.portNumber = atoi(s.c_str() + 5);
            inPort = true;
            continue;
        }

        if (inPort) {
            // State: Active
            if (s.find("State:") == 0) {
                current.portState = s.substr(s.find(":") + 2);
            }
            // Rate: 200
            else if (s.find("Rate:") == 0) {
                current.rateGbps = atof(s.c_str() + 5);
            }
            // Node GUID: 0xb8cef60300123456
            else if (s.find("Node GUID:") == 0) {
                std::string guidStr = s.substr(s.find(":") + 2);
                current.guid = strtoull(guidStr.c_str(), nullptr, 16);
            }
        }
    }

    // Add last device
    if (inDevice && !current.deviceName.empty()) {
        devices.push_back(current);
    }

    pclose(fp);
    return devices;
}

/// Get GPU-NIC affinity based on NUMA topology
void detectGpuNicAffinity(std::vector<RDMAInfo>& nics,
                          const std::vector<GPUInfo>& gpus) {
    // Try to determine which GPUs have NUMA affinity to which NICs
    // This is done by checking if they're on the same NUMA node

    for (auto& nic : nics) {
        // Try to get NIC NUMA node from sysfs
        std::string numaPath = "/sys/class/infiniband/" + nic.deviceName +
                              "/device/numa_node";
        std::ifstream file(numaPath);
        int nicNuma = -1;
        if (file.is_open()) {
            file >> nicNuma;
        }

        if (nicNuma >= 0) {
            for (const auto& gpu : gpus) {
                if (gpu.numaNode == nicNuma) {
                    nic.affinityGpus.push_back(gpu.deviceId);
                }
            }
        }
    }
}

/// Check if GPUDirect RDMA is supported for a NIC
bool checkGdrSupport([[maybe_unused]] const std::string& deviceName) {
    // Check for nvidia_peermem module (device-specific checks could be added later)
    std::ifstream modules("/proc/modules");
    if (modules.is_open()) {
        std::string line;
        while (std::getline(modules, line)) {
            if (line.find("nvidia_peermem") != std::string::npos ||
                line.find("nv_peer_mem") != std::string::npos) {
                return true;
            }
        }
    }

    // Also check for gdrdrv
    std::ifstream gdr("/dev/gdrdrv");
    return gdr.good();
}

/// Get list of all InfiniBand devices from sysfs
std::vector<std::string> listIbDevices() {
    std::vector<std::string> devices;

    DIR* dir = opendir("/sys/class/infiniband");
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_name[0] != '.') {
                devices.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }

    return devices;
}

} // namespace nperf
