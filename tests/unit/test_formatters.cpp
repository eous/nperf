/**
 * Unit tests for nperf/output/formatter.h, json_formatter.h, text_formatter.h
 *
 * Tests JSON and text output formatting.
 */

#include <gtest/gtest.h>
#include "nperf/output/formatter.h"
#include "nperf/output/json_formatter.h"
#include "nperf/output/text_formatter.h"
#include "test_utils.h"
#include <nlohmann/json.hpp>

namespace nperf {
namespace testing {

// ============================================================================
// Formatter Factory Tests
// ============================================================================

class FormatterFactoryTest : public ::testing::Test {};

TEST_F(FormatterFactoryTest, CreateTextFormatter) {
    auto formatter = createFormatter(OutputFormat::Text);
    EXPECT_NE(formatter, nullptr);
}

TEST_F(FormatterFactoryTest, CreateJsonFormatter) {
    auto formatter = createFormatter(OutputFormat::JSON);
    EXPECT_NE(formatter, nullptr);
}

TEST_F(FormatterFactoryTest, CreateJsonPrettyFormatter) {
    auto formatter = createFormatter(OutputFormat::JSONPretty);
    EXPECT_NE(formatter, nullptr);
}

// ============================================================================
// JsonFormatter Tests
// ============================================================================

class JsonFormatterTest : public ::testing::Test {
protected:
    JsonFormatter formatter{true};  // Pretty print for readability
    NperfConfig config;
    TopologyInfo topology;

    void SetUp() override {
        // Set up minimal config
        config.benchmark.operation = CollectiveOp::AllReduce;
        config.benchmark.dataType = DataType::Float32;
        config.benchmark.minBytes = 1024;
        config.benchmark.maxBytes = 1024;

        // Set up minimal topology
        topology.hostname = "test-host";
        topology.ncclVersionMajor = 2;
        topology.ncclVersionMinor = 18;
        topology.ncclVersionPatch = 3;

        GPUInfo gpu;
        gpu.deviceId = 0;
        gpu.name = "Test GPU";
        topology.gpus.push_back(gpu);
    }

    bool isValidJson(const std::string& str) {
        try {
            auto json = nlohmann::json::parse(str);
            return !json.is_null();  // Use the result to avoid unused warning
        } catch (...) {
            return false;
        }
    }
};

TEST_F(JsonFormatterTest, FormatHeaderProducesValidJson) {
    std::string output = formatter.formatHeader(config, topology);
    EXPECT_TRUE(isValidJson(output)) << "Output was: " << output;
}

TEST_F(JsonFormatterTest, FormatHeaderContainsOperation) {
    std::string output = formatter.formatHeader(config, topology);
    auto json = nlohmann::json::parse(output);

    EXPECT_TRUE(json.contains("config"));
    EXPECT_TRUE(json["config"].contains("operation"));
}

TEST_F(JsonFormatterTest, FormatHeaderContainsTopology) {
    std::string output = formatter.formatHeader(config, topology);
    auto json = nlohmann::json::parse(output);

    EXPECT_TRUE(json.contains("topology"));
    EXPECT_EQ(json["topology"]["hostname"], "test-host");
}

TEST_F(JsonFormatterTest, FormatSizeResultProducesValidJson) {
    SizeResult result;
    result.messageBytes = 1024;
    result.iterations = 10;
    result.timing.avgUs = 50.0;
    result.bandwidth.busGBps = 5.0;

    std::string output = formatter.formatSizeResult(result);
    EXPECT_TRUE(isValidJson(output)) << "Output was: " << output;
}

TEST_F(JsonFormatterTest, FormatSizeResultContainsFields) {
    SizeResult result;
    result.messageBytes = 4096;
    result.iterations = 20;
    result.timing.avgUs = 100.0;
    result.timing.minUs = 90.0;
    result.timing.maxUs = 110.0;
    result.bandwidth.busGBps = 10.0;

    std::string output = formatter.formatSizeResult(result);
    auto json = nlohmann::json::parse(output);

    EXPECT_EQ(json["messageBytes"], 4096);
    EXPECT_EQ(json["iterations"], 20);
    EXPECT_TRUE(json.contains("latency"));  // JSON uses "latency" not "timing"
    EXPECT_TRUE(json.contains("bandwidth"));
}

TEST_F(JsonFormatterTest, FormatIntervalProducesValidJson) {
    IntervalReport interval;
    interval.startSeconds = 0.0;
    interval.endSeconds = 1.0;
    interval.currentBandwidthGBps = 8.5;

    std::string output = formatter.formatInterval(interval);
    EXPECT_TRUE(isValidJson(output)) << "Output was: " << output;
}

TEST_F(JsonFormatterTest, FormatResultsProducesValidJson) {
    BenchmarkResults results;
    results.config = config.benchmark;
    results.topology = topology;

    SizeResult sr;
    sr.messageBytes = 1024;
    sr.iterations = 10;
    sr.bandwidth.busGBps = 5.0;
    results.sizeResults.push_back(sr);

    std::string output = formatter.formatResults(results);
    EXPECT_TRUE(isValidJson(output)) << "Output was: " << output;
}

TEST_F(JsonFormatterTest, FormatResultsContainsSizeResults) {
    BenchmarkResults results;
    results.config = config.benchmark;
    results.topology = topology;

    for (int i = 0; i < 3; ++i) {
        SizeResult sr;
        sr.messageBytes = 1024 * (i + 1);
        sr.bandwidth.busGBps = 5.0 * (i + 1);
        results.sizeResults.push_back(sr);
    }

    std::string output = formatter.formatResults(results);
    auto json = nlohmann::json::parse(output);

    EXPECT_TRUE(json.contains("results"));
    EXPECT_EQ(json["results"].size(), 3u);
}

TEST_F(JsonFormatterTest, FormatTopologyProducesValidJson) {
    std::string output = formatter.formatTopology(topology);
    EXPECT_TRUE(isValidJson(output)) << "Output was: " << output;
}

TEST_F(JsonFormatterTest, FormatTopologyContainsGpus) {
    std::string output = formatter.formatTopology(topology);
    auto json = nlohmann::json::parse(output);

    EXPECT_TRUE(json.contains("gpus"));
    EXPECT_EQ(json["gpus"].size(), 1u);
    EXPECT_EQ(json["gpus"][0]["name"], "Test GPU");
}

TEST_F(JsonFormatterTest, NonPrettyPrintIsSingleLine) {
    JsonFormatter compactFormatter{false};

    SizeResult result;
    result.messageBytes = 1024;
    result.iterations = 10;

    std::string output = compactFormatter.formatSizeResult(result);

    // Compact JSON should have no newlines (except maybe at the end)
    size_t newlineCount = std::count(output.begin(), output.end(), '\n');
    EXPECT_LE(newlineCount, 1u);  // At most one trailing newline
}

// ============================================================================
// TextFormatter Tests
// ============================================================================

class TextFormatterTest : public ::testing::Test {
protected:
    TextFormatter formatter;
    NperfConfig config;
    TopologyInfo topology;

    void SetUp() override {
        config.benchmark.operation = CollectiveOp::AllReduce;
        config.benchmark.dataType = DataType::Float32;
        config.benchmark.minBytes = 1024;
        config.benchmark.maxBytes = 1024;

        topology.hostname = "test-host";

        GPUInfo gpu;
        gpu.deviceId = 0;
        gpu.name = "NVIDIA A100-SXM4-80GB";
        topology.gpus.push_back(gpu);
    }
};

TEST_F(TextFormatterTest, FormatHeaderNotEmpty) {
    std::string output = formatter.formatHeader(config, topology);
    EXPECT_FALSE(output.empty());
}

TEST_F(TextFormatterTest, FormatHeaderContainsOperation) {
    std::string output = formatter.formatHeader(config, topology);
    EXPECT_TRUE(contains(output, "AllReduce") || contains(output, "allreduce"));
}

TEST_F(TextFormatterTest, FormatHeaderContainsDataType) {
    std::string output = formatter.formatHeader(config, topology);
    EXPECT_TRUE(contains(output, "float32") || contains(output, "Float32"));
}

TEST_F(TextFormatterTest, FormatSizeResultNotEmpty) {
    SizeResult result;
    result.messageBytes = 1024;
    result.iterations = 10;
    result.timing.avgUs = 50.0;
    result.bandwidth.busGBps = 5.0;

    std::string output = formatter.formatSizeResult(result);
    EXPECT_FALSE(output.empty());
}

TEST_F(TextFormatterTest, FormatSizeResultContainsSize) {
    SizeResult result;
    result.messageBytes = 1024;
    result.bandwidth.busGBps = 5.0;

    std::string output = formatter.formatSizeResult(result);
    // Should contain size in some form (1024, 1 KB, etc.)
    EXPECT_TRUE(contains(output, "1024") || contains(output, "KB") || contains(output, "1 K"));
}

TEST_F(TextFormatterTest, FormatSizeResultContainsBandwidth) {
    SizeResult result;
    result.messageBytes = 1024;
    result.bandwidth.busGBps = 5.0;

    std::string output = formatter.formatSizeResult(result);
    EXPECT_TRUE(contains(output, "5") || contains(output, "GB"));
}

TEST_F(TextFormatterTest, FormatIntervalNotEmpty) {
    IntervalReport interval;
    interval.startSeconds = 0.0;
    interval.endSeconds = 1.0;
    interval.currentBandwidthGBps = 8.5;

    std::string output = formatter.formatInterval(interval);
    EXPECT_FALSE(output.empty());
}

TEST_F(TextFormatterTest, FormatResultsNotEmpty) {
    BenchmarkResults results;
    results.config = config.benchmark;
    results.topology = topology;

    SizeResult sr;
    sr.messageBytes = 1024;
    sr.bandwidth.busGBps = 5.0;
    results.sizeResults.push_back(sr);

    std::string output = formatter.formatResults(results);
    EXPECT_FALSE(output.empty());
}

TEST_F(TextFormatterTest, FormatResultsContainsTable) {
    BenchmarkResults results;
    results.config = config.benchmark;

    for (int i = 0; i < 3; ++i) {
        SizeResult sr;
        sr.messageBytes = 1024 * (1 << i);
        sr.bandwidth.busGBps = 5.0;
        results.sizeResults.push_back(sr);
    }

    std::string output = formatter.formatResults(results);

    // Should contain separators or table-like structure
    EXPECT_TRUE(contains(output, "-") || contains(output, "|") || contains(output, "="));
}

TEST_F(TextFormatterTest, FormatTopologyNotEmpty) {
    std::string output = formatter.formatTopology(topology);
    EXPECT_FALSE(output.empty());
}

TEST_F(TextFormatterTest, FormatTopologyContainsGpuName) {
    std::string output = formatter.formatTopology(topology);
    EXPECT_TRUE(contains(output, "A100") || contains(output, "GPU"));
}

TEST_F(TextFormatterTest, FormatTopologyContainsHostname) {
    std::string output = formatter.formatTopology(topology);
    EXPECT_TRUE(contains(output, "test-host") || contains(output, "Host"));
}

// ============================================================================
// Empty/Edge Case Tests
// ============================================================================

class FormatterEdgeCasesTest : public ::testing::Test {
protected:
    JsonFormatter jsonFormatter{true};
    TextFormatter textFormatter;
};

TEST_F(FormatterEdgeCasesTest, EmptyTopology) {
    TopologyInfo topology;  // All defaults, no GPUs

    std::string jsonOutput = jsonFormatter.formatTopology(topology);
    EXPECT_FALSE(jsonOutput.empty());

    // Should still be valid JSON
    EXPECT_NO_THROW({ auto json = nlohmann::json::parse(jsonOutput); (void)json; });
}

TEST_F(FormatterEdgeCasesTest, EmptyResults) {
    BenchmarkResults results;  // No size results

    std::string jsonOutput = jsonFormatter.formatResults(results);
    EXPECT_FALSE(jsonOutput.empty());

    // Should still be valid JSON
    EXPECT_NO_THROW({ auto json = nlohmann::json::parse(jsonOutput); (void)json; });
}

TEST_F(FormatterEdgeCasesTest, ZeroBandwidth) {
    SizeResult result;
    result.messageBytes = 0;
    result.bandwidth.busGBps = 0.0;
    result.bandwidth.dataGBps = 0.0;

    std::string jsonOutput = jsonFormatter.formatSizeResult(result);
    EXPECT_FALSE(jsonOutput.empty());

    auto json = nlohmann::json::parse(jsonOutput);
    EXPECT_EQ(json["messageBytes"], 0);
}

TEST_F(FormatterEdgeCasesTest, VerificationFailed) {
    SizeResult result;
    result.messageBytes = 1024;
    result.verified = false;
    result.verifyErrors = 100;

    std::string jsonOutput = jsonFormatter.formatSizeResult(result);
    auto json = nlohmann::json::parse(jsonOutput);

    EXPECT_FALSE(json["verified"]);
}

TEST_F(FormatterEdgeCasesTest, LargeMessageSize) {
    SizeResult result;
    result.messageBytes = 1024ULL * 1024 * 1024 * 10;  // 10 GB
    result.bandwidth.busGBps = 100.0;

    std::string jsonOutput = jsonFormatter.formatSizeResult(result);
    auto json = nlohmann::json::parse(jsonOutput);

    EXPECT_EQ(json["messageBytes"], 1024ULL * 1024 * 1024 * 10);
}

// ============================================================================
// P2P Matrix Formatting Tests (TextFormatter)
// ============================================================================

class TextFormatterP2PMatrixTest : public ::testing::Test {
protected:
    TextFormatter formatter;
    TopologyInfo topology;

    void SetUp() override {
        // Create 4 GPUs with P2P matrix
        for (int i = 0; i < 4; ++i) {
            GPUInfo gpu;
            gpu.deviceId = i;
            gpu.name = "GPU " + std::to_string(i);
            topology.gpus.push_back(gpu);
        }

        // Initialize P2P matrix
        topology.p2pMatrix.resize(4);
        for (int i = 0; i < 4; ++i) {
            topology.p2pMatrix[i].resize(4);
            for (int j = 0; j < 4; ++j) {
                P2PInfo info;
                info.gpu1 = i;
                info.gpu2 = j;
                if (i == j) {
                    info.linkType = LinkType::Same;
                } else {
                    info.linkType = LinkType::NVLink;
                    info.nvlinkLanes = 12;
                }
                info.accessSupported = true;
                topology.p2pMatrix[i][j] = info;
            }
        }
    }
};

TEST_F(TextFormatterP2PMatrixTest, FormatTopologyContainsMatrix) {
    std::string output = formatter.formatTopology(topology);

    // Should contain GPU identifiers
    EXPECT_TRUE(contains(output, "GPU") || contains(output, "0") || contains(output, "1"));
}

TEST_F(TextFormatterP2PMatrixTest, FormatTopologyContainsLinkTypes) {
    std::string output = formatter.formatTopology(topology);

    // Should contain link type indicators
    // "X" for same, "NV" for NVLink, etc.
    EXPECT_TRUE(contains(output, "X") || contains(output, "NV") || contains(output, "Same"));
}

}  // namespace testing
}  // namespace nperf
