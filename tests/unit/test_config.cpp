/**
 * Unit tests for nperf/config.h
 *
 * Tests configuration parsing, validation, and size formatting.
 */

#include <gtest/gtest.h>
#include "nperf/config.h"
#include "test_utils.h"

namespace nperf {
namespace testing {

// ============================================================================
// parseSize Tests
// ============================================================================

class ParseSizeTest : public ::testing::Test {};

TEST_F(ParseSizeTest, ParsePlainNumbers) {
    EXPECT_EQ(parseSize("0"), 0u);
    EXPECT_EQ(parseSize("1"), 1u);
    EXPECT_EQ(parseSize("1024"), 1024u);
    EXPECT_EQ(parseSize("1000000"), 1000000u);
}

TEST_F(ParseSizeTest, ParseWithKiloSuffix) {
    EXPECT_EQ(parseSize("1K"), 1024u);
    EXPECT_EQ(parseSize("1k"), 1024u);
    EXPECT_EQ(parseSize("4K"), 4096u);
    EXPECT_EQ(parseSize("1024K"), 1024u * 1024);
}

TEST_F(ParseSizeTest, ParseWithMegaSuffix) {
    EXPECT_EQ(parseSize("1M"), 1024u * 1024);
    EXPECT_EQ(parseSize("1m"), 1024u * 1024);
    EXPECT_EQ(parseSize("16M"), 16u * 1024 * 1024);
}

TEST_F(ParseSizeTest, ParseWithGigaSuffix) {
    EXPECT_EQ(parseSize("1G"), 1024ULL * 1024 * 1024);
    EXPECT_EQ(parseSize("1g"), 1024ULL * 1024 * 1024);
    EXPECT_EQ(parseSize("4G"), 4ULL * 1024 * 1024 * 1024);
}

TEST_F(ParseSizeTest, ParseWithTeraSuffix) {
    EXPECT_EQ(parseSize("1T"), 1024ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(parseSize("1t"), 1024ULL * 1024 * 1024 * 1024);
}

TEST_F(ParseSizeTest, ParseEmpty) {
    EXPECT_EQ(parseSize(""), 0u);
}

TEST_F(ParseSizeTest, ParseInvalidReturnsZero) {
    EXPECT_EQ(parseSize("abc"), 0u);
    EXPECT_EQ(parseSize("K"), 0u);
    // Unknown suffix - parser extracts numeric prefix, ignores rest
    EXPECT_EQ(parseSize("1X"), 1u);  // Parses "1", ignores "X"
}

TEST_F(ParseSizeTest, ParseNegativeNumbers) {
    // Negative numbers: stoull wraps around for size_t
    // The implementation logs a warning about overflow for -1K, -100M
    // -1 parses to SIZE_MAX (unsigned wrap)
    size_t result = parseSize("-1");
    // Either large (wrap around) or 0 if overflow detected
    EXPECT_TRUE(result == 0u || result > 1000000000000ULL);

    // -1K and -100M should return 0 due to overflow detection
    EXPECT_EQ(parseSize("-1K"), 0u);
    EXPECT_EQ(parseSize("-100M"), 0u);
}

TEST_F(ParseSizeTest, ParseWithWhitespace) {
    // std::stoull handles leading whitespace, so these parse successfully
    EXPECT_EQ(parseSize("  1024  "), 1024u);
    EXPECT_EQ(parseSize(" 1K"), 1024u);
}

TEST_F(ParseSizeTest, ParseVeryLargeValues) {
    // Test large but valid values
    EXPECT_EQ(parseSize("1T"), 1024ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(parseSize("16T"), 16ULL * 1024 * 1024 * 1024 * 1024);
}

// ============================================================================
// formatSize Tests
// ============================================================================

class FormatSizeTest : public ::testing::Test {};

TEST_F(FormatSizeTest, FormatBytes) {
    EXPECT_EQ(formatSize(0), "0 B");
    EXPECT_EQ(formatSize(1), "1 B");
    EXPECT_EQ(formatSize(512), "512 B");
    EXPECT_EQ(formatSize(1023), "1023 B");
}

TEST_F(FormatSizeTest, FormatKilobytes) {
    EXPECT_EQ(formatSize(1024), "1 KB");
    EXPECT_EQ(formatSize(2048), "2 KB");
    EXPECT_EQ(formatSize(1024 + 512), "1.50 KB");
}

TEST_F(FormatSizeTest, FormatMegabytes) {
    EXPECT_EQ(formatSize(1024 * 1024), "1 MB");
    EXPECT_EQ(formatSize(16 * 1024 * 1024), "16 MB");
}

TEST_F(FormatSizeTest, FormatGigabytes) {
    EXPECT_EQ(formatSize(1024ULL * 1024 * 1024), "1 GB");
    EXPECT_EQ(formatSize(8ULL * 1024 * 1024 * 1024), "8 GB");
}

TEST_F(FormatSizeTest, FormatTerabytes) {
    EXPECT_EQ(formatSize(1024ULL * 1024 * 1024 * 1024), "1 TB");
}

// ============================================================================
// BenchmarkConfig::getMessageSizes Tests
// ============================================================================

class GetMessageSizesTest : public ::testing::Test {};

TEST_F(GetMessageSizesTest, SingleSize) {
    BenchmarkConfig config;
    config.minBytes = 1024;
    config.maxBytes = 1024;
    config.stepFactor = 2;

    auto sizes = config.getMessageSizes();
    ASSERT_EQ(sizes.size(), 1u);
    EXPECT_EQ(sizes[0], 1024u);
}

TEST_F(GetMessageSizesTest, StepFactor2) {
    BenchmarkConfig config;
    config.minBytes = 1024;
    config.maxBytes = 8192;
    config.stepFactor = 2;

    auto sizes = config.getMessageSizes();
    // Expected: 1024, 2048, 4096, 8192
    ASSERT_EQ(sizes.size(), 4u);
    EXPECT_EQ(sizes[0], 1024u);
    EXPECT_EQ(sizes[1], 2048u);
    EXPECT_EQ(sizes[2], 4096u);
    EXPECT_EQ(sizes[3], 8192u);
}

TEST_F(GetMessageSizesTest, StepFactor4) {
    BenchmarkConfig config;
    config.minBytes = 1024;
    config.maxBytes = 16384;
    config.stepFactor = 4;

    auto sizes = config.getMessageSizes();
    // Expected: 1024, 4096, 16384
    ASSERT_EQ(sizes.size(), 3u);
    EXPECT_EQ(sizes[0], 1024u);
    EXPECT_EQ(sizes[1], 4096u);
    EXPECT_EQ(sizes[2], 16384u);
}

TEST_F(GetMessageSizesTest, IncludesMaxIfNotExactMultiple) {
    BenchmarkConfig config;
    config.minBytes = 1024;
    config.maxBytes = 5000;  // Not a power of 2
    config.stepFactor = 2;

    auto sizes = config.getMessageSizes();
    // Expected: 1024, 2048, 4096, 5000 (maxBytes added at end)
    ASSERT_GE(sizes.size(), 2u);
    EXPECT_EQ(sizes.front(), 1024u);
    EXPECT_EQ(sizes.back(), 5000u);
}

TEST_F(GetMessageSizesTest, ZeroMinBytes) {
    BenchmarkConfig config;
    config.minBytes = 0;
    config.maxBytes = 1024;
    config.stepFactor = 2;

    auto sizes = config.getMessageSizes();
    ASSERT_GE(sizes.size(), 1u);
    EXPECT_EQ(sizes.front(), 1u);  // Minimum is 1 for 0 input
}

TEST_F(GetMessageSizesTest, StepFactor1HandledGracefully) {
    BenchmarkConfig config;
    config.minBytes = 1024;
    config.maxBytes = 4096;
    config.stepFactor = 1;  // Would cause infinite loop if not handled

    auto sizes = config.getMessageSizes();
    // Should return min and max only
    ASSERT_GE(sizes.size(), 1u);
}

// ============================================================================
// NperfConfig::validate Tests
// ============================================================================

class ConfigValidateTest : public ::testing::Test {
protected:
    NperfConfig config;
    std::string error;
};

TEST_F(ConfigValidateTest, ValidDefaultConfig) {
    // Default config should be valid (local mode, iteration-based)
    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, InvalidMinGreaterThanMax) {
    config.benchmark.minBytes = 2048;
    config.benchmark.maxBytes = 1024;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "minBytes"));
}

TEST_F(ConfigValidateTest, InvalidZeroMinBytes) {
    config.benchmark.minBytes = 0;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "minBytes"));
}

TEST_F(ConfigValidateTest, InvalidZeroIterations) {
    config.benchmark.iterations = 0;
    config.benchmark.useTimeBased = false;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "iterations"));
}

TEST_F(ConfigValidateTest, InvalidNegativeIterations) {
    config.benchmark.iterations = -1;
    config.benchmark.useTimeBased = false;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "iterations"));
}

TEST_F(ConfigValidateTest, TimeBasedModeValid) {
    config.benchmark.useTimeBased = true;
    config.benchmark.testDurationSeconds = 10.0;
    config.benchmark.iterations = 0;  // Ignored in time-based mode

    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, InvalidTimeBasedWithZeroDuration) {
    config.benchmark.useTimeBased = true;
    config.benchmark.testDurationSeconds = 0.0;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "testDurationSeconds"));
}

TEST_F(ConfigValidateTest, InvalidTimeBasedWithNegativeDuration) {
    config.benchmark.useTimeBased = true;
    config.benchmark.testDurationSeconds = -1.0;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "testDurationSeconds"));
}

TEST_F(ConfigValidateTest, SocketClientWithoutHost) {
    config.coordination.mode = CoordinationMode::Socket;
    config.coordination.isServer = false;
    config.coordination.serverHost = "";

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "serverHost"));
}

TEST_F(ConfigValidateTest, SocketServerValid) {
    config.coordination.mode = CoordinationMode::Socket;
    config.coordination.isServer = true;
    // serverHost not required for server

    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, SocketClientWithHostValid) {
    config.coordination.mode = CoordinationMode::Socket;
    config.coordination.isServer = false;
    config.coordination.serverHost = "localhost";

    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, NcclBootstrapWithoutRank) {
    config.coordination.mode = CoordinationMode::NcclBootstrap;
    config.coordination.rank = -1;
    config.coordination.worldSize = 2;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "rank"));
}

TEST_F(ConfigValidateTest, NcclBootstrapWithoutWorldSize) {
    config.coordination.mode = CoordinationMode::NcclBootstrap;
    config.coordination.rank = 0;
    config.coordination.worldSize = -1;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "world-size"));
}

TEST_F(ConfigValidateTest, NcclBootstrapRankExceedsWorldSize) {
    config.coordination.mode = CoordinationMode::NcclBootstrap;
    config.coordination.rank = 5;
    config.coordination.worldSize = 4;

    EXPECT_FALSE(config.validate(error));
    EXPECT_TRUE(contains(error, "rank"));
}

TEST_F(ConfigValidateTest, NcclBootstrapValid) {
    config.coordination.mode = CoordinationMode::NcclBootstrap;
    config.coordination.rank = 0;
    config.coordination.worldSize = 4;

    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, LocalModeValid) {
    config.coordination.mode = CoordinationMode::Local;
    // No additional requirements

    EXPECT_TRUE(config.validate(error)) << error;
}

TEST_F(ConfigValidateTest, MPIModeValid) {
    config.coordination.mode = CoordinationMode::MPI;
    // MPI mode doesn't require additional validation
    // (MPI itself handles initialization)

    EXPECT_TRUE(config.validate(error)) << error;
}

// ============================================================================
// BenchmarkConfig Default Values Tests
// ============================================================================

class BenchmarkConfigDefaultsTest : public ::testing::Test {};

TEST_F(BenchmarkConfigDefaultsTest, DefaultOperation) {
    BenchmarkConfig config;
    EXPECT_EQ(config.operation, CollectiveOp::AllReduce);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultDataType) {
    BenchmarkConfig config;
    EXPECT_EQ(config.dataType, DataType::Float32);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultReduceOp) {
    BenchmarkConfig config;
    EXPECT_EQ(config.reduceOp, ReduceOp::Sum);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultAlgorithm) {
    BenchmarkConfig config;
    EXPECT_EQ(config.algorithm, Algorithm::Auto);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultProtocol) {
    BenchmarkConfig config;
    EXPECT_EQ(config.protocol, Protocol::Auto);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultSizes) {
    BenchmarkConfig config;
    EXPECT_EQ(config.minBytes, 1024u);
    EXPECT_EQ(config.maxBytes, 1024u);
    EXPECT_EQ(config.stepFactor, 2u);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultDuration) {
    BenchmarkConfig config;
    EXPECT_FALSE(config.useTimeBased);
    EXPECT_EQ(config.iterations, 20);
    EXPECT_DOUBLE_EQ(config.testDurationSeconds, 0.0);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultWarmup) {
    BenchmarkConfig config;
    EXPECT_EQ(config.warmupIterations, 5);
    EXPECT_DOUBLE_EQ(config.omitSeconds, 0.0);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultVerification) {
    BenchmarkConfig config;
    EXPECT_EQ(config.verifyMode, VerifyMode::None);
    EXPECT_DOUBLE_EQ(config.verifyTolerance, 1e-5);
}

TEST_F(BenchmarkConfigDefaultsTest, DefaultCudaOptions) {
    BenchmarkConfig config;
    EXPECT_FALSE(config.useCudaGraph);
    EXPECT_EQ(config.cudaDevice, -1);
}

// ============================================================================
// CoordinationConfig Default Values Tests
// ============================================================================

class CoordinationConfigDefaultsTest : public ::testing::Test {};

TEST_F(CoordinationConfigDefaultsTest, DefaultMode) {
    CoordinationConfig config;
    EXPECT_EQ(config.mode, CoordinationMode::Local);
}

TEST_F(CoordinationConfigDefaultsTest, DefaultLocalSettings) {
    CoordinationConfig config;
    EXPECT_EQ(config.numLocalGpus, -1);  // All available
}

TEST_F(CoordinationConfigDefaultsTest, DefaultSocketSettings) {
    CoordinationConfig config;
    EXPECT_TRUE(config.serverHost.empty());
    EXPECT_EQ(config.port, 5201);
    EXPECT_FALSE(config.isServer);
    EXPECT_EQ(config.expectedClients, 1);
}

TEST_F(CoordinationConfigDefaultsTest, DefaultNcclBootstrapSettings) {
    CoordinationConfig config;
    EXPECT_EQ(config.rank, -1);
    EXPECT_EQ(config.worldSize, -1);
}

// ============================================================================
// OutputConfig Default Values Tests
// ============================================================================

class OutputConfigDefaultsTest : public ::testing::Test {};

TEST_F(OutputConfigDefaultsTest, DefaultFormat) {
    OutputConfig config;
    EXPECT_EQ(config.format, OutputFormat::Text);
}

TEST_F(OutputConfigDefaultsTest, DefaultOutputFile) {
    OutputConfig config;
    EXPECT_TRUE(config.outputFile.empty());  // stdout
}

TEST_F(OutputConfigDefaultsTest, DefaultTopologySettings) {
    OutputConfig config;
    EXPECT_FALSE(config.showTopology);
    EXPECT_EQ(config.topoFormat, TopoFormat::Matrix);
    EXPECT_FALSE(config.topologyOnly);
    EXPECT_FALSE(config.showTransport);
}

TEST_F(OutputConfigDefaultsTest, DefaultVerbosity) {
    OutputConfig config;
    EXPECT_FALSE(config.verbose);
    EXPECT_FALSE(config.debug);
}

}  // namespace testing
}  // namespace nperf
