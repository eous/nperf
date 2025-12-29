/**
 * Unit tests for nperf/core/metrics.h
 *
 * Tests MetricsCalculator, effectiveCount, and countDescription.
 */

#include <gtest/gtest.h>
#include "nperf/core/metrics.h"
#include "test_utils.h"

namespace nperf {
namespace testing {

// ============================================================================
// MetricsCalculator Tests
// ============================================================================

class MetricsCalculatorTest : public ::testing::Test {
protected:
    MetricsCalculator calc;
};

TEST_F(MetricsCalculatorTest, DefaultConstruction) {
    // Should not crash
    auto latencies = calc.getLatencies(1024);
    EXPECT_TRUE(latencies.empty());
}

TEST_F(MetricsCalculatorTest, AddSingleSample) {
    calc.addSample(1024, 10.0);

    auto latencies = calc.getLatencies(1024);
    ASSERT_EQ(latencies.size(), 1u);
    EXPECT_DOUBLE_EQ(latencies[0], 10.0);
}

TEST_F(MetricsCalculatorTest, AddMultipleSamples) {
    calc.addSample(1024, 10.0);
    calc.addSample(1024, 20.0);
    calc.addSample(1024, 15.0);

    auto latencies = calc.getLatencies(1024);
    ASSERT_EQ(latencies.size(), 3u);
    EXPECT_DOUBLE_EQ(latencies[0], 10.0);
    EXPECT_DOUBLE_EQ(latencies[1], 20.0);
    EXPECT_DOUBLE_EQ(latencies[2], 15.0);
}

TEST_F(MetricsCalculatorTest, MultipleSizes) {
    calc.addSample(1024, 10.0);
    calc.addSample(2048, 20.0);
    calc.addSample(1024, 15.0);
    calc.addSample(2048, 25.0);

    auto latencies1k = calc.getLatencies(1024);
    auto latencies2k = calc.getLatencies(2048);

    ASSERT_EQ(latencies1k.size(), 2u);
    ASSERT_EQ(latencies2k.size(), 2u);

    EXPECT_DOUBLE_EQ(latencies1k[0], 10.0);
    EXPECT_DOUBLE_EQ(latencies1k[1], 15.0);
    EXPECT_DOUBLE_EQ(latencies2k[0], 20.0);
    EXPECT_DOUBLE_EQ(latencies2k[1], 25.0);
}

TEST_F(MetricsCalculatorTest, GetLatenciesForUnknownSize) {
    calc.addSample(1024, 10.0);

    auto latencies = calc.getLatencies(2048);  // Never added
    EXPECT_TRUE(latencies.empty());
}

TEST_F(MetricsCalculatorTest, ClearRemovesAllSamples) {
    calc.addSample(1024, 10.0);
    calc.addSample(2048, 20.0);

    calc.clear();

    EXPECT_TRUE(calc.getLatencies(1024).empty());
    EXPECT_TRUE(calc.getLatencies(2048).empty());
}

TEST_F(MetricsCalculatorTest, ComputeResultSingleSample) {
    calc.setOperation(CollectiveOp::AllReduce);
    calc.setWorldSize(4);
    calc.addSample(1024, 10.0);  // 10 us latency

    SizeResult result = calc.computeResult(1024);

    EXPECT_EQ(result.messageBytes, 1024u);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_DOUBLE_EQ(result.timing.avgUs, 10.0);
    EXPECT_DOUBLE_EQ(result.timing.minUs, 10.0);
    EXPECT_DOUBLE_EQ(result.timing.maxUs, 10.0);
}

TEST_F(MetricsCalculatorTest, ComputeResultMultipleSamples) {
    calc.setOperation(CollectiveOp::Broadcast);
    calc.setWorldSize(2);

    calc.addSample(1024, 10.0);
    calc.addSample(1024, 20.0);
    calc.addSample(1024, 30.0);

    SizeResult result = calc.computeResult(1024);

    EXPECT_EQ(result.iterations, 3);
    EXPECT_DOUBLE_EQ(result.timing.avgUs, 20.0);
    EXPECT_DOUBLE_EQ(result.timing.minUs, 10.0);
    EXPECT_DOUBLE_EQ(result.timing.maxUs, 30.0);
}

TEST_F(MetricsCalculatorTest, ComputeResultBandwidth) {
    calc.setOperation(CollectiveOp::Broadcast);
    calc.setWorldSize(2);

    // 1 GB in 1 second = 1 GB/s
    size_t bytes = 1024ULL * 1024 * 1024;  // 1 GB
    double latencyUs = 1000000.0;          // 1 second

    calc.addSample(bytes, latencyUs);

    SizeResult result = calc.computeResult(bytes);

    // Data bandwidth should be approximately 1 GB/s
    EXPECT_GT(result.bandwidth.dataGBps, 0.9);
    EXPECT_LT(result.bandwidth.dataGBps, 1.1);
}

TEST_F(MetricsCalculatorTest, ComputeResultNoSamples) {
    calc.setOperation(CollectiveOp::AllReduce);
    calc.setWorldSize(4);

    SizeResult result = calc.computeResult(1024);

    EXPECT_EQ(result.messageBytes, 1024u);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_DOUBLE_EQ(result.timing.avgUs, 0.0);
}

// ============================================================================
// effectiveCount Tests
// ============================================================================

class EffectiveCountTest : public ::testing::Test {};

TEST_F(EffectiveCountTest, AllReduceFullCount) {
    // AllReduce uses the full count
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;  // 4 bytes per element
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::AllReduce, bytes, dtype, worldSize);
    EXPECT_EQ(count, 4096u / 4);  // 1024 elements
}

TEST_F(EffectiveCountTest, AllGatherDividedByWorldSize) {
    // AllGather: each rank contributes bytes/worldSize
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;  // 4 bytes per element
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::AllGather, bytes, dtype, worldSize);
    EXPECT_EQ(count, (4096u / 4) / 4);  // 256 elements per rank
}

TEST_F(EffectiveCountTest, ReduceScatterDividedByWorldSize) {
    // ReduceScatter: each rank receives bytes/worldSize
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;  // 4 bytes per element
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::ReduceScatter, bytes, dtype, worldSize);
    EXPECT_EQ(count, (4096u / 4) / 4);  // 256 elements per rank
}

TEST_F(EffectiveCountTest, GatherDividedByWorldSize) {
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::Gather, bytes, dtype, worldSize);
    EXPECT_EQ(count, (4096u / 4) / 4);  // 256 elements per rank
}

TEST_F(EffectiveCountTest, ScatterDividedByWorldSize) {
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::Scatter, bytes, dtype, worldSize);
    EXPECT_EQ(count, (4096u / 4) / 4);  // 256 elements per rank
}

TEST_F(EffectiveCountTest, BroadcastFullCount) {
    size_t bytes = 4096;
    DataType dtype = DataType::Float32;
    int worldSize = 4;

    size_t count = effectiveCount(CollectiveOp::Broadcast, bytes, dtype, worldSize);
    EXPECT_EQ(count, 4096u / 4);  // Full count
}

TEST_F(EffectiveCountTest, DifferentDataTypes) {
    size_t bytes = 1024;

    // Float32: 4 bytes
    EXPECT_EQ(effectiveCount(CollectiveOp::AllReduce, bytes, DataType::Float32, 1), 256u);

    // Float64: 8 bytes
    EXPECT_EQ(effectiveCount(CollectiveOp::AllReduce, bytes, DataType::Float64, 1), 128u);

    // Float16: 2 bytes
    EXPECT_EQ(effectiveCount(CollectiveOp::AllReduce, bytes, DataType::Float16, 1), 512u);

    // Int8: 1 byte
    EXPECT_EQ(effectiveCount(CollectiveOp::AllReduce, bytes, DataType::Int8, 1), 1024u);
}

// ============================================================================
// countDescription Tests
// ============================================================================

class CountDescriptionTest : public ::testing::Test {};

TEST_F(CountDescriptionTest, AllGatherReturnsSendcount) {
    EXPECT_STREQ(countDescription(CollectiveOp::AllGather), "sendcount");
}

TEST_F(CountDescriptionTest, ReduceScatterReturnsRecvcount) {
    EXPECT_STREQ(countDescription(CollectiveOp::ReduceScatter), "recvcount");
}

TEST_F(CountDescriptionTest, GatherReturnsSendcount) {
    EXPECT_STREQ(countDescription(CollectiveOp::Gather), "sendcount");
}

TEST_F(CountDescriptionTest, ScatterReturnsRecvcount) {
    EXPECT_STREQ(countDescription(CollectiveOp::Scatter), "recvcount");
}

TEST_F(CountDescriptionTest, OtherOpsReturnCount) {
    EXPECT_STREQ(countDescription(CollectiveOp::AllReduce), "count");
    EXPECT_STREQ(countDescription(CollectiveOp::Broadcast), "count");
    EXPECT_STREQ(countDescription(CollectiveOp::Reduce), "count");
    EXPECT_STREQ(countDescription(CollectiveOp::AlltoAll), "count");
    EXPECT_STREQ(countDescription(CollectiveOp::SendRecv), "count");
}

}  // namespace testing
}  // namespace nperf
