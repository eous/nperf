// PCIe topology detection utilities

#include "nperf/topology/detector.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>

namespace nperf {

/// Parse PCI bus ID string into components
struct PciBusId {
    int domain = 0;
    int bus = 0;
    int device = 0;
    int function = 0;

    bool parse(const std::string& str) {
        // Format: DDDD:BB:DD.F or BB:DD.F
        unsigned int udomain, ubus, udevice, ufunction;
        int count = sscanf(str.c_str(), "%x:%x:%x.%x",
                          &udomain, &ubus, &udevice, &ufunction);
        if (count == 4) {
            domain = static_cast<int>(udomain);
            bus = static_cast<int>(ubus);
            device = static_cast<int>(udevice);
            function = static_cast<int>(ufunction);
            return true;
        }

        // Try without domain
        domain = 0;
        count = sscanf(str.c_str(), "%x:%x.%x", &ubus, &udevice, &ufunction);
        if (count == 3) {
            bus = static_cast<int>(ubus);
            device = static_cast<int>(udevice);
            function = static_cast<int>(ufunction);
            return true;
        }
        return false;
    }

    std::string toString() const {
        char buf[32];
        snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%d",
                domain, bus, device, function);
        return std::string(buf);
    }
};

/// Get NUMA node for a GPU
int getGpuNumaNode(int deviceId) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess) {
        return -1;
    }

    // Construct sysfs path
    char pciBusId[20];
    snprintf(pciBusId, sizeof(pciBusId), "%04x:%02x:%02x.0",
            props.pciDomainID, props.pciBusID, props.pciDeviceID);

    std::string numaPath = "/sys/bus/pci/devices/" +
                          std::string(pciBusId) + "/numa_node";

    std::ifstream file(numaPath);
    if (file.is_open()) {
        int numa;
        file >> numa;
        return numa;
    }

    return -1;
}

/// Check if two GPUs share a PCIe switch
bool sharesPcieSwitch(int gpu1, int gpu2) {
    cudaDeviceProp props1, props2;
    if (cudaGetDeviceProperties(&props1, gpu1) != cudaSuccess ||
        cudaGetDeviceProperties(&props2, gpu2) != cudaSuccess) {
        return false;
    }

    // Simple heuristic: same bus number implies same switch
    return props1.pciDomainID == props2.pciDomainID &&
           props1.pciBusID == props2.pciBusID;
}

/// Check if two GPUs are on the same NUMA node
bool sameNumaNode(int gpu1, int gpu2) {
    int numa1 = getGpuNumaNode(gpu1);
    int numa2 = getGpuNumaNode(gpu2);

    if (numa1 < 0 || numa2 < 0) {
        return false; // Unknown
    }

    return numa1 == numa2;
}

/// Estimate PCIe bandwidth between two GPUs
double estimatePcieBandwidth(int gpu1, [[maybe_unused]] int gpu2) {
    // PCIe Gen4 x16: ~32 GB/s bidirectional
    // PCIe Gen5 x16: ~64 GB/s bidirectional

    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, gpu1) != cudaSuccess) {
        return 16.0; // Conservative estimate
    }

    // Check compute capability for generation hint
    // Ampere+ typically has PCIe Gen4
    if (props.major >= 8) {
        return 32.0; // Gen4
    }
    return 16.0; // Gen3
}

} // namespace nperf
