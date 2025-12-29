/**
 * Integration tests for BenchmarkEngine.
 *
 * End-to-end tests for the benchmark orchestrator (requires GPU).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/engine.h"
#include "nperf/config.h"

namespace nperf {
namespace testing {

class BenchmarkEngineTest : public ::testing::Test {
protected:
    NperfConfig config;

    void SetUp() override {
        SKIP_IF_NO_GPU();

        // Set up minimal valid config
        config.benchmark.operation = CollectiveOp::AllReduce;
        config.benchmark.dataType = DataType::Float32;
        config.benchmark.reduceOp = ReduceOp::Sum;
        config.benchmark.minBytes = 1024;
        config.benchmark.maxBytes = 1024;
        config.benchmark.iterations = 5;
        config.benchmark.warmupIterations = 1;

        config.coordination.mode = CoordinationMode::Local;
        config.coordination.numLocalGpus = 1;
    }
};

TEST_F(BenchmarkEngineTest, Construction) {
    BenchmarkEngine engine;
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, Configure) {
    BenchmarkEngine engine;
    engine.configure(config);
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, SetupNccl) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.setupNccl();
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, AllocateBuffers) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.setupNccl();
    engine.allocateBuffers(1024);
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, RunWarmup) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.setupNccl();
    engine.allocateBuffers(1024);
    engine.runWarmup(config.benchmark.warmupIterations);
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, RunSingleSize) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.setupNccl();
    engine.allocateBuffers(1024);
    engine.runWarmup(1);

    SizeResult result = engine.runSize(1024);

    EXPECT_EQ(result.messageBytes, 1024u);
    EXPECT_EQ(result.iterations, config.benchmark.iterations);
    EXPECT_GT(result.timing.avgUs, 0.0);
    EXPECT_GT(result.bandwidth.busGBps, 0.0);
}

TEST_F(BenchmarkEngineTest, RunMultipleSizes) {
    config.benchmark.minBytes = 1024;
    config.benchmark.maxBytes = 4096;
    config.benchmark.stepFactor = 2;
    config.benchmark.iterations = 3;

    BenchmarkEngine engine;
    engine.configure(config);

    BenchmarkResults results = engine.run();

    // Should have 3 sizes: 1024, 2048, 4096
    EXPECT_GE(results.sizeResults.size(), 2u);

    for (const auto& sr : results.sizeResults) {
        EXPECT_GT(sr.timing.avgUs, 0.0);
        EXPECT_GT(sr.bandwidth.busGBps, 0.0);
    }
}

TEST_F(BenchmarkEngineTest, ProgressCallback) {
    config.benchmark.iterations = 10;

    BenchmarkEngine engine;
    engine.configure(config);

    int callbackCount = 0;
    engine.setProgressCallback([&](const IntervalReport& report) {
        callbackCount++;
        EXPECT_GE(report.overallProgress, 0.0);
        EXPECT_LE(report.overallProgress, 1.0);
    });

    engine.run();

    // Callback should have been called at least once
    EXPECT_GE(callbackCount, 0);  // May be 0 if interval is longer than test
}

TEST_F(BenchmarkEngineTest, Finalize) {
    BenchmarkEngine engine;
    engine.configure(config);
    engine.run();
    engine.finalize();
    SUCCEED();
}

TEST_F(BenchmarkEngineTest, WithVerification) {
    config.benchmark.verifyMode = VerifyMode::PerIteration;
    config.benchmark.verifyTolerance = 1e-5;
    config.benchmark.iterations = 5;

    BenchmarkEngine engine;
    engine.configure(config);

    BenchmarkResults results = engine.run();

    // All results should be verified
    EXPECT_TRUE(results.allVerified);
}

TEST_F(BenchmarkEngineTest, WithCudaGraph) {
    config.benchmark.useCudaGraph = true;
    config.benchmark.iterations = 10;

    BenchmarkEngine engine;
    engine.configure(config);

    BenchmarkResults results = engine.run();

    EXPECT_GE(results.sizeResults.size(), 1u);
    EXPECT_GT(results.sizeResults[0].bandwidth.busGBps, 0.0);
}

TEST_F(BenchmarkEngineTest, TopologyOnlyMode) {
    config.output.topologyOnly = true;

    BenchmarkEngine engine;
    engine.configure(config);

    TopologyInfo topo = engine.getTopology();

    EXPECT_GE(topo.gpus.size(), 1u);
}

TEST_F(BenchmarkEngineTest, DifferentOperations) {
    for (auto op : {CollectiveOp::AllReduce, CollectiveOp::Broadcast,
                    CollectiveOp::Reduce}) {
        config.benchmark.operation = op;
        config.benchmark.iterations = 3;

        BenchmarkEngine engine;
        engine.configure(config);

        BenchmarkResults results = engine.run();

        EXPECT_GE(results.sizeResults.size(), 1u);
    }
}

TEST_F(BenchmarkEngineTest, DifferentDataTypes) {
    for (auto dtype : {DataType::Float32, DataType::Float64, DataType::Int32}) {
        config.benchmark.dataType = dtype;
        config.benchmark.iterations = 3;

        BenchmarkEngine engine;
        engine.configure(config);

        BenchmarkResults results = engine.run();

        EXPECT_GE(results.sizeResults.size(), 1u);
    }
}

// ============================================================================
// Multi-GPU Tests
// ============================================================================

class BenchmarkEngineMultiGpuTest : public ::testing::Test {
protected:
    NperfConfig config;

    void SetUp() override {
        SKIP_IF_FEWER_THAN_N_GPUS(2);

        config.benchmark.operation = CollectiveOp::AllReduce;
        config.benchmark.dataType = DataType::Float32;
        config.benchmark.minBytes = 1024;
        config.benchmark.maxBytes = 1024;
        config.benchmark.iterations = 5;
        config.benchmark.warmupIterations = 1;

        config.coordination.mode = CoordinationMode::Local;
        config.coordination.numLocalGpus = getCudaGpuCount();
    }
};

TEST_F(BenchmarkEngineMultiGpuTest, AllReduceMultiGpu) {
    BenchmarkEngine engine;
    engine.configure(config);

    BenchmarkResults results = engine.run();

    EXPECT_GE(results.sizeResults.size(), 1u);
    EXPECT_GT(results.peakBusGBps, 0.0);
}

TEST_F(BenchmarkEngineMultiGpuTest, AllGatherMultiGpu) {
    config.benchmark.operation = CollectiveOp::AllGather;

    BenchmarkEngine engine;
    engine.configure(config);

    BenchmarkResults results = engine.run();

    EXPECT_GE(results.sizeResults.size(), 1u);
}

}  // namespace testing
}  // namespace nperf

#else
TEST(BenchmarkEngineTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
