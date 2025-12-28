#include "nperf/output/json_formatter.h"
#include "nperf/version.h"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace nperf {

using json = nlohmann::json;

JsonFormatter::JsonFormatter(bool prettyPrint)
    : prettyPrint_(prettyPrint) {
}

std::string JsonFormatter::formatHeader(const NperfConfig& config,
                                        const TopologyInfo& topology) {
    json j;
    j["version"] = NPERF_VERSION;
    j["config"] = configToJson(config.benchmark);
    j["topology"] = topologyToJson(topology);

    if (prettyPrint_) {
        return j.dump(2);
    }
    return j.dump();
}

std::string JsonFormatter::formatSizeResult(const SizeResult& result) {
    json j = sizeResultToJson(result);
    if (prettyPrint_) {
        return j.dump(2);
    }
    return j.dump();
}

std::string JsonFormatter::formatInterval(const IntervalReport& interval) {
    json j = intervalToJson(interval);
    if (prettyPrint_) {
        return j.dump(2);
    }
    return j.dump();
}

std::string JsonFormatter::formatResults(const BenchmarkResults& results) {
    json j = resultsToJson(results);
    if (prettyPrint_) {
        return j.dump(2);
    }
    return j.dump();
}

std::string JsonFormatter::formatTopology(const TopologyInfo& topology) {
    json j = topologyToJson(topology);
    if (prettyPrint_) {
        return j.dump(2);
    }
    return j.dump();
}

json JsonFormatter::configToJson(const BenchmarkConfig& config) {
    json j;
    j["operation"] = collectiveOpName(config.operation);
    j["dataType"] = dataTypeName(config.dataType);
    j["reduceOp"] = reduceOpName(config.reduceOp);
    j["algorithm"] = algorithmName(config.algorithm);
    j["protocol"] = protocolName(config.protocol);
    j["minBytes"] = config.minBytes;
    j["maxBytes"] = config.maxBytes;
    j["stepFactor"] = config.stepFactor;
    j["iterations"] = config.iterations;
    j["warmupIterations"] = config.warmupIterations;
    j["useCudaGraph"] = config.useCudaGraph;
    j["verifyMode"] = static_cast<int>(config.verifyMode);
    return j;
}

json JsonFormatter::topologyToJson(const TopologyInfo& topology) {
    json j;
    j["hostname"] = topology.hostname;
    j["ncclVersion"] = std::to_string(topology.ncclVersionMajor) + "." +
                       std::to_string(topology.ncclVersionMinor) + "." +
                       std::to_string(topology.ncclVersionPatch);
    j["gpuCount"] = topology.gpus.size();
    j["hasNVSwitch"] = topology.hasNVSwitch;

    // GPUs
    json gpuArray = json::array();
    for (const auto& gpu : topology.gpus) {
        json g;
        g["deviceId"] = gpu.deviceId;
        g["name"] = gpu.name;
        g["uuid"] = gpu.uuid;
        g["pciBusId"] = gpu.pciBusId;
        g["computeCapability"] = std::to_string(gpu.computeCapabilityMajor) + "." +
                                 std::to_string(gpu.computeCapabilityMinor);
        g["memoryGB"] = gpu.totalMemoryBytes / (1024.0 * 1024.0 * 1024.0);
        g["numaNode"] = gpu.numaNode;
        g["nvlinkCount"] = gpu.nvlinkCount;

        // NVLinks
        json nvlinks = json::array();
        for (const auto& link : gpu.nvlinks) {
            json l;
            l["linkIndex"] = link.linkIndex;
            l["targetGpu"] = link.targetGpu;
            l["version"] = link.version;
            l["active"] = link.isActive;
            nvlinks.push_back(l);
        }
        g["nvlinks"] = nvlinks;

        gpuArray.push_back(g);
    }
    j["gpus"] = gpuArray;

    // P2P Matrix
    json p2pMatrix = json::array();
    for (const auto& row : topology.p2pMatrix) {
        json rowJson = json::array();
        for (const auto& p2p : row) {
            json p;
            p["accessSupported"] = p2p.accessSupported;
            p["atomicSupported"] = p2p.atomicSupported;
            p["performanceRank"] = p2p.performanceRank;
            p["linkType"] = linkTypeLegend(p2p.linkType);
            if (p2p.linkType == LinkType::NVLink) {
                p["nvlinkLanes"] = p2p.nvlinkLanes;
                p["nvlinkVersion"] = p2p.nvlinkVersion;
            }
            rowJson.push_back(p);
        }
        p2pMatrix.push_back(rowJson);
    }
    j["p2pMatrix"] = p2pMatrix;

    // RDMA devices
    json rdmaArray = json::array();
    for (const auto& rdma : topology.rdmaDevices) {
        json r;
        r["name"] = rdma.deviceName;
        r["type"] = rdma.linkType;
        r["portState"] = rdma.portState;
        r["rateGbps"] = rdma.rateGbps;
        r["gdrSupported"] = rdma.gdrSupported;
        r["affinityGpus"] = rdma.affinityGpus;
        rdmaArray.push_back(r);
    }
    j["rdmaDevices"] = rdmaArray;

    return j;
}

json JsonFormatter::sizeResultToJson(const SizeResult& result) {
    json j;
    j["messageBytes"] = result.messageBytes;
    j["elementCount"] = result.elementCount;
    j["iterations"] = result.iterations;

    j["latency"] = {
        {"avgUs", result.timing.avgUs},
        {"minUs", result.timing.minUs},
        {"maxUs", result.timing.maxUs},
        {"stddevUs", result.timing.stddevUs},
        {"p50Us", result.timing.p50Us},
        {"p95Us", result.timing.p95Us},
        {"p99Us", result.timing.p99Us}
    };

    j["bandwidth"] = {
        {"dataGBps", result.bandwidth.dataGBps},
        {"algoGBps", result.bandwidth.algoGBps},
        {"busGBps", result.bandwidth.busGBps}
    };

    j["verified"] = result.verified;
    j["verifyErrors"] = result.verifyErrors;

    if (!result.detectedTransport.empty()) {
        j["transport"] = result.detectedTransport;
    }

    return j;
}

json JsonFormatter::intervalToJson(const IntervalReport& interval) {
    json j;
    j["start"] = interval.startSeconds;
    j["end"] = interval.endSeconds;
    j["bytesTransferred"] = interval.bytesTransferred;
    j["operationsCompleted"] = interval.operationsCompleted;
    j["bandwidthGBps"] = interval.currentBandwidthGBps;
    j["latencyUs"] = interval.currentLatencyUs;
    return j;
}

json JsonFormatter::resultsToJson(const BenchmarkResults& results) {
    json j;
    j["version"] = NPERF_VERSION;

    // Timestamp
    auto time = std::chrono::system_clock::to_time_t(results.startTime);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    j["timestamp"] = oss.str();

    j["config"] = configToJson(results.config);
    j["topology"] = topologyToJson(results.topology);

    // Results array
    json resultsArray = json::array();
    for (const auto& sr : results.sizeResults) {
        resultsArray.push_back(sizeResultToJson(sr));
    }
    j["results"] = resultsArray;

    // Intervals
    json intervalsArray = json::array();
    for (const auto& interval : results.intervals) {
        intervalsArray.push_back(intervalToJson(interval));
    }
    j["intervals"] = intervalsArray;

    // Summary
    j["summary"] = {
        {"peakBusGBps", results.peakBusGBps},
        {"avgBusGBps", results.avgBusGBps},
        {"totalBytes", results.totalBytes},
        {"totalTimeSeconds", results.totalTimeSeconds},
        {"totalIterations", results.totalIterations},
        {"verified", results.allVerified},
        {"verifyErrors", results.totalVerifyErrors}
    };

    j["rank"] = results.rank;
    j["worldSize"] = results.worldSize;

    return j;
}

} // namespace nperf
