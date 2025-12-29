/**
 * Unit tests for nperf/results.h
 *
 * Tests TimingStats::compute, getBusBandwidthFactor, computeBandwidth,
 * and BenchmarkResults::computeSummary.
 */

#include <gtest/gtest.h>
#include "nperf/results.h"
#include "test_utils.h"
#include <cmath>
#include <algorithm>

namespace nperf {
namespace testing {

// ============================================================================
// TimingStats::compute Tests
// ============================================================================

class TimingStatsComputeTest : public ::testing::Test {};

TEST_F(TimingStatsComputeTest, EmptyVector) {
    std::vector<double> latencies;
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 0);
    EXPECT_DOUBLE_EQ(stats.avgUs, 0.0);
    EXPECT_DOUBLE_EQ(stats.minUs, 0.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 0.0);
    EXPECT_DOUBLE_EQ(stats.stddevUs, 0.0);
}

TEST_F(TimingStatsComputeTest, SingleElement) {
    std::vector<double> latencies = {100.0};
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 1);
    EXPECT_DOUBLE_EQ(stats.avgUs, 100.0);
    EXPECT_DOUBLE_EQ(stats.minUs, 100.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 100.0);
    EXPECT_DOUBLE_EQ(stats.p50Us, 100.0);
    EXPECT_DOUBLE_EQ(stats.p95Us, 100.0);
    EXPECT_DOUBLE_EQ(stats.p99Us, 100.0);
    EXPECT_DOUBLE_EQ(stats.stddevUs, 0.0);
}

TEST_F(TimingStatsComputeTest, TwoElements) {
    std::vector<double> latencies = {100.0, 200.0};
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 2);
    EXPECT_DOUBLE_EQ(stats.avgUs, 150.0);
    EXPECT_DOUBLE_EQ(stats.minUs, 100.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 200.0);
}

TEST_F(TimingStatsComputeTest, ThreeElements) {
    std::vector<double> latencies = {100.0, 150.0, 200.0};
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 3);
    EXPECT_DOUBLE_EQ(stats.avgUs, 150.0);
    EXPECT_DOUBLE_EQ(stats.minUs, 100.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 200.0);
    EXPECT_DOUBLE_EQ(stats.p50Us, 150.0);  // Middle element
}

TEST_F(TimingStatsComputeTest, UnsortedInput) {
    std::vector<double> latencies = {200.0, 100.0, 150.0, 300.0, 50.0};
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 5);
    EXPECT_DOUBLE_EQ(stats.minUs, 50.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 300.0);
    // Average = (200+100+150+300+50)/5 = 160
    EXPECT_DOUBLE_EQ(stats.avgUs, 160.0);
}

TEST_F(TimingStatsComputeTest, StandardDeviationAccuracy) {
    // Use a known set where we can calculate stddev by hand
    // [1, 2, 3, 4, 5] -> mean = 3, variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
    //                         = (4 + 1 + 0 + 1 + 4) / 5 = 2
    //                  -> stddev = sqrt(2) ≈ 1.414
    std::vector<double> latencies = {1.0, 2.0, 3.0, 4.0, 5.0};
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_DOUBLE_EQ(stats.avgUs, 3.0);
    EXPECT_APPROX_EQ(stats.stddevUs, std::sqrt(2.0));
}

TEST_F(TimingStatsComputeTest, AllSameValues) {
    std::vector<double> latencies(100, 42.0);
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 100);
    EXPECT_DOUBLE_EQ(stats.avgUs, 42.0);
    EXPECT_DOUBLE_EQ(stats.minUs, 42.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 42.0);
    EXPECT_DOUBLE_EQ(stats.stddevUs, 0.0);
    EXPECT_DOUBLE_EQ(stats.p50Us, 42.0);
    EXPECT_DOUBLE_EQ(stats.p95Us, 42.0);
    EXPECT_DOUBLE_EQ(stats.p99Us, 42.0);
}

TEST_F(TimingStatsComputeTest, PercentilesWithManyElements) {
    // Create 100 elements: 1, 2, 3, ..., 100
    std::vector<double> latencies;
    for (int i = 1; i <= 100; ++i) {
        latencies.push_back(static_cast<double>(i));
    }

    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 100);
    EXPECT_DOUBLE_EQ(stats.minUs, 1.0);
    EXPECT_DOUBLE_EQ(stats.maxUs, 100.0);

    // p50 should be around 50
    EXPECT_GE(stats.p50Us, 49.0);
    EXPECT_LE(stats.p50Us, 51.0);

    // p95 should be around 95
    EXPECT_GE(stats.p95Us, 94.0);
    EXPECT_LE(stats.p95Us, 96.0);

    // p99 should be around 99
    EXPECT_GE(stats.p99Us, 98.0);
    EXPECT_LE(stats.p99Us, 100.0);
}

TEST_F(TimingStatsComputeTest, LargeVector) {
    std::vector<double> latencies(10000, 1.0);
    TimingStats stats = TimingStats::compute(latencies);

    EXPECT_EQ(stats.sampleCount, 10000);
    EXPECT_DOUBLE_EQ(stats.avgUs, 1.0);
}

// ============================================================================
// getBusBandwidthFactor Tests
// ============================================================================

class GetBusBandwidthFactorTest : public ::testing::Test {};

TEST_F(GetBusBandwidthFactorTest, WorldSize1ReturnsOne) {
    // With 1 GPU, no actual communication happens
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllReduce, 1), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllGather, 1), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Broadcast, 1), 1.0);
}

TEST_F(GetBusBandwidthFactorTest, AllReduce) {
    // AllReduce: 2 * (n-1)/n
    // For n=2: 2 * (2-1)/2 = 2 * 0.5 = 1.0
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllReduce, 2), 1.0);

    // For n=4: 2 * (4-1)/4 = 2 * 0.75 = 1.5
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllReduce, 4), 1.5);

    // For n=8: 2 * (8-1)/8 = 2 * 0.875 = 1.75
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllReduce, 8), 1.75);

    // For large n, approaches 2.0
    EXPECT_APPROX_EQ(getBusBandwidthFactor(CollectiveOp::AllReduce, 1000), 1.998);
}

TEST_F(GetBusBandwidthFactorTest, AllGather) {
    // AllGather: (n-1)/n
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllGather, 2), 0.5);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllGather, 4), 0.75);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AllGather, 8), 0.875);
}

TEST_F(GetBusBandwidthFactorTest, ReduceScatter) {
    // ReduceScatter: (n-1)/n
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::ReduceScatter, 2), 0.5);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::ReduceScatter, 4), 0.75);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::ReduceScatter, 8), 0.875);
}

TEST_F(GetBusBandwidthFactorTest, AlltoAll) {
    // AlltoAll: (n-1)/n
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AlltoAll, 2), 0.5);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::AlltoAll, 4), 0.75);
}

TEST_F(GetBusBandwidthFactorTest, GatherAndScatter) {
    // Gather/Scatter: (n-1)/n
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Gather, 4), 0.75);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Scatter, 4), 0.75);
}

TEST_F(GetBusBandwidthFactorTest, BroadcastAndReduce) {
    // Broadcast/Reduce: 1.0 (data moves once)
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Broadcast, 2), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Broadcast, 8), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Reduce, 2), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::Reduce, 8), 1.0);
}

TEST_F(GetBusBandwidthFactorTest, SendRecv) {
    // SendRecv: 1.0
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::SendRecv, 2), 1.0);
    EXPECT_DOUBLE_EQ(getBusBandwidthFactor(CollectiveOp::SendRecv, 8), 1.0);
}

TEST_F(GetBusBandwidthFactorTest, VeryLargeWorldSize) {
    // Test with typical large cluster sizes
    // AllReduce: 2 * (n-1)/n should approach 2.0 as n increases
    double factor10k = getBusBandwidthFactor(CollectiveOp::AllReduce, 10000);
    EXPECT_LT(factor10k, 2.0);
    EXPECT_GT(factor10k, 1.99);  // 2 * 9999/10000 = 1.9998

    double factor100k = getBusBandwidthFactor(CollectiveOp::AllReduce, 100000);
    EXPECT_LT(factor100k, 2.0);
    EXPECT_GT(factor100k, 1.999);

    // Verify no overflow at very large values
    double factorMillion = getBusBandwidthFactor(CollectiveOp::AllReduce, 1000000);
    EXPECT_LT(factorMillion, 2.0);
    EXPECT_GT(factorMillion, 1.9999);

    // AllGather/ReduceScatter: (n-1)/n should approach 1.0
    double allGatherFactor = getBusBandwidthFactor(CollectiveOp::AllGather, 100000);
    EXPECT_LT(allGatherFactor, 1.0);
    EXPECT_GT(allGatherFactor, 0.999);
}

// ============================================================================
// computeBandwidth Tests
// ============================================================================

class ComputeBandwidthTest : public ::testing::Test {};

TEST_F(ComputeBandwidthTest, ZeroLatencyReturnsZero) {
    BandwidthMetrics bw = computeBandwidth(1024, 0.0, CollectiveOp::AllReduce, 2);

    EXPECT_DOUBLE_EQ(bw.dataGBps, 0.0);
    EXPECT_DOUBLE_EQ(bw.algoGBps, 0.0);
    EXPECT_DOUBLE_EQ(bw.busGBps, 0.0);
}

TEST_F(ComputeBandwidthTest, NegativeLatencyReturnsZero) {
    BandwidthMetrics bw = computeBandwidth(1024, -1.0, CollectiveOp::AllReduce, 2);

    EXPECT_DOUBLE_EQ(bw.dataGBps, 0.0);
    EXPECT_DOUBLE_EQ(bw.algoGBps, 0.0);
    EXPECT_DOUBLE_EQ(bw.busGBps, 0.0);
}

TEST_F(ComputeBandwidthTest, BasicCalculation) {
    // 1 GB in 1 second = 1 GB/s
    // 1 GB = 1073741824 bytes
    // 1 second = 1,000,000 microseconds
    size_t bytes = 1024ULL * 1024 * 1024;  // 1 GB
    double latencyUs = 1000000.0;          // 1 second

    BandwidthMetrics bw = computeBandwidth(bytes, latencyUs, CollectiveOp::Broadcast, 2);

    EXPECT_APPROX_EQ(bw.dataGBps, 1.0);
    EXPECT_APPROX_EQ(bw.algoGBps, 1.0);
    EXPECT_APPROX_EQ(bw.busGBps, 1.0);  // Broadcast factor is 1.0
}

TEST_F(ComputeBandwidthTest, BusBandwidthIncludesFactor) {
    // Use AllReduce which has factor 2*(n-1)/n
    // For n=2: factor = 1.0
    // For n=4: factor = 1.5
    size_t bytes = 1024ULL * 1024 * 1024;  // 1 GB
    double latencyUs = 1000000.0;          // 1 second

    BandwidthMetrics bw2 = computeBandwidth(bytes, latencyUs, CollectiveOp::AllReduce, 2);
    BandwidthMetrics bw4 = computeBandwidth(bytes, latencyUs, CollectiveOp::AllReduce, 4);

    EXPECT_APPROX_EQ(bw2.dataGBps, 1.0);
    EXPECT_APPROX_EQ(bw2.busGBps, 1.0);  // factor = 1.0 for n=2

    EXPECT_APPROX_EQ(bw4.dataGBps, 1.0);
    EXPECT_APPROX_EQ(bw4.busGBps, 1.5);  // factor = 1.5 for n=4
}

TEST_F(ComputeBandwidthTest, SmallMessage) {
    // 1 KB in 10 us = 0.1 GB/s
    size_t bytes = 1024;
    double latencyUs = 10.0;

    BandwidthMetrics bw = computeBandwidth(bytes, latencyUs, CollectiveOp::Broadcast, 2);

    // 1024 bytes / 10us = 1024 / (10 * 1e-6) bytes/s = 102.4 MB/s = 0.1 GB/s
    // Actually: 1024 / (10 * 1e-6) / (1024^3) = 1024 / 10e-6 / 1073741824
    //         = 1024 * 1e5 / 1073741824 ≈ 0.0000953674 GB/s
    // Wait, let me recalculate:
    // 1024 bytes / (10 * 1e-6 seconds) = 1024 / 0.00001 = 102,400,000 bytes/s
    // = 102.4 MB/s = 0.1 GB/s (approximately)
    EXPECT_GT(bw.dataGBps, 0.09);
    EXPECT_LT(bw.dataGBps, 0.11);
}

// ============================================================================
// BenchmarkResults::computeSummary Tests
// ============================================================================

class ComputeSummaryTest : public ::testing::Test {
protected:
    BenchmarkResults results;

    void SetUp() override {
        results.sizeResults.clear();
        results.intervals.clear();
    }
};

TEST_F(ComputeSummaryTest, EmptyResults) {
    results.computeSummary();

    // Should not crash, values remain at defaults
    EXPECT_DOUBLE_EQ(results.peakBusGBps, 0.0);
    EXPECT_DOUBLE_EQ(results.avgBusGBps, 0.0);
}

TEST_F(ComputeSummaryTest, SingleSizeResult) {
    SizeResult sr;
    sr.messageBytes = 1024;
    sr.iterations = 10;
    sr.bandwidth.busGBps = 5.0;
    sr.verified = true;
    sr.verifyErrors = 0;
    results.sizeResults.push_back(sr);

    results.computeSummary();

    EXPECT_DOUBLE_EQ(results.peakBusGBps, 5.0);
    EXPECT_DOUBLE_EQ(results.avgBusGBps, 5.0);
    EXPECT_DOUBLE_EQ(results.totalBytes, 1024.0 * 10);
    EXPECT_EQ(results.totalIterations, 10);
    EXPECT_TRUE(results.allVerified);
    EXPECT_EQ(results.totalVerifyErrors, 0);
}

TEST_F(ComputeSummaryTest, MultipleSizeResults) {
    SizeResult sr1;
    sr1.messageBytes = 1024;
    sr1.iterations = 10;
    sr1.bandwidth.busGBps = 2.0;
    sr1.verified = true;
    results.sizeResults.push_back(sr1);

    SizeResult sr2;
    sr2.messageBytes = 4096;
    sr2.iterations = 5;
    sr2.bandwidth.busGBps = 8.0;
    sr2.verified = true;
    results.sizeResults.push_back(sr2);

    SizeResult sr3;
    sr3.messageBytes = 8192;
    sr3.iterations = 3;
    sr3.bandwidth.busGBps = 6.0;
    sr3.verified = true;
    results.sizeResults.push_back(sr3);

    results.computeSummary();

    EXPECT_DOUBLE_EQ(results.peakBusGBps, 8.0);
    EXPECT_APPROX_EQ(results.avgBusGBps, (2.0 + 8.0 + 6.0) / 3.0);
    EXPECT_DOUBLE_EQ(results.totalBytes, 1024.0 * 10 + 4096.0 * 5 + 8192.0 * 3);
    EXPECT_EQ(results.totalIterations, 10 + 5 + 3);
    EXPECT_TRUE(results.allVerified);
}

TEST_F(ComputeSummaryTest, VerificationFailure) {
    SizeResult sr1;
    sr1.verified = true;
    sr1.verifyErrors = 0;
    sr1.bandwidth.busGBps = 5.0;
    results.sizeResults.push_back(sr1);

    SizeResult sr2;
    sr2.verified = false;
    sr2.verifyErrors = 3;
    sr2.bandwidth.busGBps = 4.0;
    results.sizeResults.push_back(sr2);

    results.computeSummary();

    EXPECT_FALSE(results.allVerified);
    EXPECT_EQ(results.totalVerifyErrors, 3);
}

TEST_F(ComputeSummaryTest, MultipleVerificationErrors) {
    SizeResult sr1;
    sr1.verified = false;
    sr1.verifyErrors = 2;
    sr1.bandwidth.busGBps = 5.0;
    results.sizeResults.push_back(sr1);

    SizeResult sr2;
    sr2.verified = false;
    sr2.verifyErrors = 5;
    sr2.bandwidth.busGBps = 4.0;
    results.sizeResults.push_back(sr2);

    results.computeSummary();

    EXPECT_FALSE(results.allVerified);
    EXPECT_EQ(results.totalVerifyErrors, 7);
}

TEST_F(ComputeSummaryTest, TimeCalculation) {
    results.startTime = std::chrono::system_clock::now();
    results.endTime = results.startTime + std::chrono::milliseconds(5000);

    // Add a size result so computeSummary doesn't early return
    SizeResult sr;
    sr.bandwidth.busGBps = 1.0;
    results.sizeResults.push_back(sr);

    results.computeSummary();

    EXPECT_APPROX_EQ(results.totalTimeSeconds, 5.0);
}

// ============================================================================
// SizeResult Default Values Tests
// ============================================================================

class SizeResultDefaultsTest : public ::testing::Test {};

TEST_F(SizeResultDefaultsTest, DefaultValues) {
    SizeResult sr;

    EXPECT_EQ(sr.messageBytes, 0u);
    EXPECT_EQ(sr.elementCount, 0u);
    EXPECT_EQ(sr.iterations, 0);
    EXPECT_TRUE(sr.verified);
    EXPECT_EQ(sr.verifyErrors, 0);
    EXPECT_TRUE(sr.detectedTransport.empty());
}

// ============================================================================
// IntervalReport Default Values Tests
// ============================================================================

class IntervalReportDefaultsTest : public ::testing::Test {};

TEST_F(IntervalReportDefaultsTest, DefaultValues) {
    IntervalReport ir;

    EXPECT_DOUBLE_EQ(ir.startSeconds, 0.0);
    EXPECT_DOUBLE_EQ(ir.endSeconds, 0.0);
    EXPECT_EQ(ir.bytesTransferred, 0u);
    EXPECT_EQ(ir.operationsCompleted, 0);
    EXPECT_DOUBLE_EQ(ir.currentBandwidthGBps, 0.0);
    EXPECT_DOUBLE_EQ(ir.currentLatencyUs, 0.0);
    EXPECT_EQ(ir.currentSizeIndex, 0u);
    EXPECT_EQ(ir.totalSizes, 0u);
    EXPECT_EQ(ir.currentMessageBytes, 0u);
    EXPECT_EQ(ir.currentIteration, 0);
    EXPECT_EQ(ir.totalIterations, 0);
    EXPECT_DOUBLE_EQ(ir.overallProgress, 0.0);
}

// ============================================================================
// IterationResult Default Values Tests
// ============================================================================

class IterationResultDefaultsTest : public ::testing::Test {};

TEST_F(IterationResultDefaultsTest, DefaultValues) {
    IterationResult ir;

    EXPECT_DOUBLE_EQ(ir.latencyUs, 0.0);
    EXPECT_EQ(ir.messageBytes, 0u);
    EXPECT_TRUE(ir.verified);
    EXPECT_TRUE(ir.verifyError.empty());
}

}  // namespace testing
}  // namespace nperf
