/**
 * Unit tests for nperf/cli/parser.h
 *
 * Tests command line argument parsing via ArgParser.
 */

#include <gtest/gtest.h>
#include "nperf/cli/parser.h"
#include "test_utils.h"
#include <vector>
#include <string>
#include <cstring>

namespace nperf {
namespace testing {

// Helper to create argv from vector of strings
class ArgvBuilder {
public:
    ArgvBuilder& add(const std::string& arg) {
        args_.push_back(arg);
        return *this;
    }

    int argc() const { return static_cast<int>(ptrs_.size()); }

    /**
     * Get argv array. Rebuilds ptrs_ each call to ensure pointers are valid.
     * Safe to call multiple times - previous pointers become invalid but
     * new array reflects current args_ state.
     */
    char** argv() {
        ptrs_.clear();
        for (auto& s : args_) {
            ptrs_.push_back(const_cast<char*>(s.c_str()));
        }
        return ptrs_.data();
    }

private:
    std::vector<std::string> args_;
    std::vector<char*> ptrs_;
};

// ============================================================================
// Basic Parsing Tests
// ============================================================================

class ArgParserBasicTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");  // Program name
    }
};

TEST_F(ArgParserBasicTest, DefaultsWithNoArgs) {
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    const auto& config = parser.config();
    EXPECT_EQ(config.coordination.mode, CoordinationMode::Local);
    EXPECT_EQ(config.benchmark.operation, CollectiveOp::AllReduce);
    EXPECT_EQ(config.benchmark.dataType, DataType::Float32);
    EXPECT_EQ(config.output.format, OutputFormat::Text);
}

TEST_F(ArgParserBasicTest, HelpFlagShort) {
    args.add("-h");
    bool result = parser.parse(args.argc(), args.argv());

    EXPECT_FALSE(result);
    EXPECT_TRUE(parser.helpRequested());
}

TEST_F(ArgParserBasicTest, HelpFlagLong) {
    args.add("--help");
    bool result = parser.parse(args.argc(), args.argv());

    EXPECT_FALSE(result);
    EXPECT_TRUE(parser.helpRequested());
}

// ============================================================================
// Message Size Tests
// ============================================================================

class ArgParserSizeTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserSizeTest, MinBytesKilobytes) {
    args.add("-b").add("4K");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.minBytes, 4096u);
}

TEST_F(ArgParserSizeTest, MinBytesMegabytes) {
    args.add("-b").add("16M");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.minBytes, 16u * 1024 * 1024);
}

TEST_F(ArgParserSizeTest, MinBytesGigabytes) {
    args.add("-b").add("1G");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.minBytes, 1024ULL * 1024 * 1024);
}

TEST_F(ArgParserSizeTest, MaxBytesDefaultsToMin) {
    args.add("-b").add("8K");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.minBytes, 8192u);
    EXPECT_EQ(parser.config().benchmark.maxBytes, 8192u);
}

TEST_F(ArgParserSizeTest, MinAndMaxBytes) {
    args.add("-b").add("1K").add("-B").add("1G");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.minBytes, 1024u);
    EXPECT_EQ(parser.config().benchmark.maxBytes, 1024ULL * 1024 * 1024);
}

TEST_F(ArgParserSizeTest, StepFactor) {
    args.add("-S").add("4");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.stepFactor, 4u);
}

// ============================================================================
// Collective Operation Tests
// ============================================================================

class ArgParserOperationTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserOperationTest, AllReduce) {
    args.add("--op").add("allreduce");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::AllReduce);
}

TEST_F(ArgParserOperationTest, AllGather) {
    args.add("--op").add("allgather");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::AllGather);
}

TEST_F(ArgParserOperationTest, Broadcast) {
    args.add("--op").add("broadcast");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::Broadcast);
}

TEST_F(ArgParserOperationTest, Reduce) {
    args.add("--op").add("reduce");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::Reduce);
}

TEST_F(ArgParserOperationTest, ReduceScatter) {
    args.add("--op").add("reducescatter");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::ReduceScatter);
}

TEST_F(ArgParserOperationTest, AlltoAll) {
    args.add("--op").add("alltoall");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::AlltoAll);
}

TEST_F(ArgParserOperationTest, SendRecv) {
    args.add("--op").add("sendrecv");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.operation, CollectiveOp::SendRecv);
}

TEST_F(ArgParserOperationTest, InvalidOperation) {
    args.add("--op").add("invalid");
    bool result = parser.parse(args.argc(), args.argv());

    EXPECT_FALSE(result);
    EXPECT_FALSE(parser.errorMessage().empty());
}

// ============================================================================
// Data Type Tests
// ============================================================================

class ArgParserDtypeTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserDtypeTest, Float32) {
    args.add("--dtype").add("float32");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.dataType, DataType::Float32);
}

TEST_F(ArgParserDtypeTest, Float64) {
    args.add("--dtype").add("float64");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.dataType, DataType::Float64);
}

TEST_F(ArgParserDtypeTest, Float16) {
    args.add("--dtype").add("float16");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.dataType, DataType::Float16);
}

TEST_F(ArgParserDtypeTest, BFloat16) {
    args.add("--dtype").add("bfloat16");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.dataType, DataType::BFloat16);
}

TEST_F(ArgParserDtypeTest, Int32) {
    args.add("--dtype").add("int32");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.dataType, DataType::Int32);
}

// ============================================================================
// Reduce Operation Tests
// ============================================================================

class ArgParserRedOpTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserRedOpTest, Sum) {
    args.add("--redop").add("sum");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.reduceOp, ReduceOp::Sum);
}

TEST_F(ArgParserRedOpTest, Prod) {
    args.add("--redop").add("prod");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.reduceOp, ReduceOp::Prod);
}

TEST_F(ArgParserRedOpTest, Min) {
    args.add("--redop").add("min");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.reduceOp, ReduceOp::Min);
}

TEST_F(ArgParserRedOpTest, Max) {
    args.add("--redop").add("max");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.reduceOp, ReduceOp::Max);
}

TEST_F(ArgParserRedOpTest, Avg) {
    args.add("--redop").add("avg");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.reduceOp, ReduceOp::Avg);
}

// ============================================================================
// Coordination Mode Tests
// ============================================================================

class ArgParserCoordinationTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserCoordinationTest, LocalMode) {
    args.add("--local");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().coordination.mode, CoordinationMode::Local);
}

TEST_F(ArgParserCoordinationTest, MpiMode) {
    args.add("--mpi");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().coordination.mode, CoordinationMode::MPI);
}

TEST_F(ArgParserCoordinationTest, SocketServerMode) {
    args.add("-s");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().coordination.mode, CoordinationMode::Socket);
    EXPECT_TRUE(parser.config().coordination.isServer);
}

TEST_F(ArgParserCoordinationTest, SocketClientMode) {
    args.add("-c").add("localhost");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().coordination.mode, CoordinationMode::Socket);
    EXPECT_FALSE(parser.config().coordination.isServer);
    EXPECT_EQ(parser.config().coordination.serverHost, "localhost");
}

TEST_F(ArgParserCoordinationTest, NcclBootstrapMode) {
    args.add("--nccl-bootstrap").add("--rank").add("0").add("--world-size").add("4");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().coordination.mode, CoordinationMode::NcclBootstrap);
    EXPECT_EQ(parser.config().coordination.rank, 0);
    EXPECT_EQ(parser.config().coordination.worldSize, 4);
}

TEST_F(ArgParserCoordinationTest, NcclBootstrapWithoutRankFails) {
    args.add("--nccl-bootstrap").add("--world-size").add("4");
    bool result = parser.parse(args.argc(), args.argv());

    EXPECT_FALSE(result);
    EXPECT_TRUE(contains(parser.errorMessage(), "rank"));
}

TEST_F(ArgParserCoordinationTest, NcclBootstrapWithoutWorldSizeFails) {
    args.add("--nccl-bootstrap").add("--rank").add("0");
    bool result = parser.parse(args.argc(), args.argv());

    EXPECT_FALSE(result);
    EXPECT_TRUE(contains(parser.errorMessage(), "world-size"));
}

// ============================================================================
// Output Tests
// ============================================================================

class ArgParserOutputTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserOutputTest, JsonOutput) {
    args.add("-J");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().output.format, OutputFormat::JSONPretty);
}

TEST_F(ArgParserOutputTest, OutputFile) {
    args.add("-o").add("output.json");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().output.outputFile, "output.json");
}

TEST_F(ArgParserOutputTest, TopologyOnly) {
    args.add("--topology");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_TRUE(parser.config().output.topologyOnly);
}

TEST_F(ArgParserOutputTest, TopoFormatMatrix) {
    args.add("--topology").add("--topo-format").add("matrix");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().output.topoFormat, TopoFormat::Matrix);
}

TEST_F(ArgParserOutputTest, TopoFormatDot) {
    args.add("--topology").add("--topo-format").add("dot");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().output.topoFormat, TopoFormat::DOT);
}

TEST_F(ArgParserOutputTest, TopoFormatJson) {
    args.add("--topology").add("--topo-format").add("json");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().output.topoFormat, TopoFormat::JSON);
}

TEST_F(ArgParserOutputTest, Verbose) {
    args.add("-v");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_TRUE(parser.config().output.verbose);
}

TEST_F(ArgParserOutputTest, Debug) {
    args.add("--debug");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_TRUE(parser.config().output.debug);
}

// ============================================================================
// CUDA Options Tests
// ============================================================================

class ArgParserCudaTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserCudaTest, CudaGraph) {
    args.add("--graph");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_TRUE(parser.config().benchmark.useCudaGraph);
}

TEST_F(ArgParserCudaTest, CudaDevice) {
    args.add("--device").add("2");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.cudaDevice, 2);
}

TEST_F(ArgParserCudaTest, CudaDeviceNonNumeric) {
    args.add("--device").add("abc");
    bool result = parser.parse(args.argc(), args.argv());
    // CLI11 should reject non-numeric device values
    EXPECT_FALSE(result);
}

TEST_F(ArgParserCudaTest, CudaDeviceNegative) {
    args.add("--device").add("-1");
    bool result = parser.parse(args.argc(), args.argv());
    // Negative device values should be rejected or handled gracefully
    // (depends on implementation - either fail or accept as signed int)
    // We accept the parse, but document behavior - runtime validation
    // will catch invalid devices
    if (result) {
        // If parsed as signed, value may be -1
        EXPECT_LT(parser.config().benchmark.cudaDevice, 0);
    }
}

// ============================================================================
// Verification Tests
// ============================================================================

class ArgParserVerifyTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserVerifyTest, VerifyEnabled) {
    args.add("--verify");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.verifyMode, VerifyMode::PerIteration);
}

TEST_F(ArgParserVerifyTest, VerifyTolerance) {
    args.add("--verify").add("--verify-tolerance").add("1e-3");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_DOUBLE_EQ(parser.config().benchmark.verifyTolerance, 1e-3);
}

// ============================================================================
// Duration Tests
// ============================================================================

class ArgParserDurationTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserDurationTest, Iterations) {
    args.add("-i").add("50");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.iterations, 50);
    EXPECT_FALSE(parser.config().benchmark.useTimeBased);
}

TEST_F(ArgParserDurationTest, TimeBased) {
    args.add("-t").add("30");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_TRUE(parser.config().benchmark.useTimeBased);
    EXPECT_DOUBLE_EQ(parser.config().benchmark.testDurationSeconds, 30.0);
}

TEST_F(ArgParserDurationTest, Warmup) {
    args.add("-w").add("10");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.warmupIterations, 10);
}

TEST_F(ArgParserDurationTest, Omit) {
    args.add("-O").add("2.5");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_DOUBLE_EQ(parser.config().benchmark.omitSeconds, 2.5);
}

// ============================================================================
// Algorithm and Protocol Tests
// ============================================================================

class ArgParserAlgoProtoTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserAlgoProtoTest, AlgorithmRing) {
    args.add("--algo").add("ring");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.algorithm, Algorithm::Ring);
}

TEST_F(ArgParserAlgoProtoTest, AlgorithmTree) {
    args.add("--algo").add("tree");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.algorithm, Algorithm::Tree);
}

TEST_F(ArgParserAlgoProtoTest, AlgorithmAuto) {
    args.add("--algo").add("auto");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.algorithm, Algorithm::Auto);
}

TEST_F(ArgParserAlgoProtoTest, ProtocolSimple) {
    args.add("--proto").add("simple");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.protocol, Protocol::Simple);
}

TEST_F(ArgParserAlgoProtoTest, ProtocolLL) {
    args.add("--proto").add("ll");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.protocol, Protocol::LL);
}

TEST_F(ArgParserAlgoProtoTest, ProtocolLL128) {
    args.add("--proto").add("ll128");
    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    EXPECT_EQ(parser.config().benchmark.protocol, Protocol::LL128);
}

// ============================================================================
// Combined Options Tests
// ============================================================================

class ArgParserCombinedTest : public ::testing::Test {
protected:
    ArgParser parser;
    ArgvBuilder args;

    void SetUp() override {
        args.add("nperf");
    }
};

TEST_F(ArgParserCombinedTest, FullBenchmarkConfig) {
    args.add("--local")
        .add("-n").add("8")
        .add("--op").add("allreduce")
        .add("-b").add("1K")
        .add("-B").add("1G")
        .add("-i").add("100")
        .add("-w").add("10")
        .add("--dtype").add("float64")
        .add("--redop").add("sum")
        .add("--graph")
        .add("--verify")
        .add("-J");

    bool result = parser.parse(args.argc(), args.argv());
    ASSERT_TRUE(result) << parser.errorMessage();

    const auto& cfg = parser.config();
    EXPECT_EQ(cfg.coordination.mode, CoordinationMode::Local);
    EXPECT_EQ(cfg.benchmark.operation, CollectiveOp::AllReduce);
    EXPECT_EQ(cfg.benchmark.minBytes, 1024u);
    EXPECT_EQ(cfg.benchmark.maxBytes, 1024ULL * 1024 * 1024);
    EXPECT_EQ(cfg.benchmark.iterations, 100);
    EXPECT_EQ(cfg.benchmark.warmupIterations, 10);
    EXPECT_EQ(cfg.benchmark.dataType, DataType::Float64);
    EXPECT_EQ(cfg.benchmark.reduceOp, ReduceOp::Sum);
    EXPECT_TRUE(cfg.benchmark.useCudaGraph);
    EXPECT_EQ(cfg.benchmark.verifyMode, VerifyMode::PerIteration);
    EXPECT_EQ(cfg.output.format, OutputFormat::JSONPretty);
}

}  // namespace testing
}  // namespace nperf
