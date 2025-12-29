/**
 * Integration tests for NCCL collective operations.
 *
 * Tests NcclCommunicator and CollectiveRunner (requires GPU + NCCL).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/collective.h"
#include "nperf/core/memory.h"
#include <vector>

namespace nperf {
namespace testing {

// ============================================================================
// Type Conversion Tests (no GPU needed)
// ============================================================================

class NcclTypeConversionTest : public ::testing::Test {};

TEST_F(NcclTypeConversionTest, DataTypeToNccl) {
    EXPECT_EQ(toNcclDataType(DataType::Float32), ncclFloat);
    EXPECT_EQ(toNcclDataType(DataType::Float64), ncclDouble);
    EXPECT_EQ(toNcclDataType(DataType::Float16), ncclHalf);
    EXPECT_EQ(toNcclDataType(DataType::BFloat16), ncclBfloat16);
    EXPECT_EQ(toNcclDataType(DataType::Int32), ncclInt32);
    EXPECT_EQ(toNcclDataType(DataType::Int64), ncclInt64);
}

TEST_F(NcclTypeConversionTest, ReduceOpToNccl) {
    EXPECT_EQ(toNcclRedOp(ReduceOp::Sum), ncclSum);
    EXPECT_EQ(toNcclRedOp(ReduceOp::Prod), ncclProd);
    EXPECT_EQ(toNcclRedOp(ReduceOp::Min), ncclMin);
    EXPECT_EQ(toNcclRedOp(ReduceOp::Max), ncclMax);
    EXPECT_EQ(toNcclRedOp(ReduceOp::Avg), ncclAvg);
}

// ============================================================================
// NcclCommunicator Tests
// ============================================================================

class NcclCommunicatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(NcclCommunicatorTest, InitAllSingleGpu) {
    NcclCommunicator comm;
    comm.initAll(1);

    EXPECT_EQ(comm.getWorldSize(), 1);
    EXPECT_EQ(comm.getRank(), 0);
}

TEST_F(NcclCommunicatorTest, InitAllMultipleGpus) {
    int gpuCount = getCudaGpuCount();
    if (gpuCount < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs for multi-GPU test";
    }

    NcclCommunicator comm;
    comm.initAll(gpuCount);

    EXPECT_EQ(comm.getWorldSize(), gpuCount);
}

TEST_F(NcclCommunicatorTest, MoveConstruction) {
    NcclCommunicator comm1;
    comm1.initAll(1);

    NcclCommunicator comm2(std::move(comm1));

    EXPECT_EQ(comm2.getWorldSize(), 1);
}

// ============================================================================
// CollectiveRunner Tests
// ============================================================================

class CollectiveRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(CollectiveRunnerTest, AllReduceSingleGpu) {
    NcclCommunicator comm;
    comm.initAll(1);

    MemoryManager manager;
    DeviceBuffer send = manager.allocateSendBuffer(1024 * sizeof(float));
    DeviceBuffer recv = manager.allocateRecvBuffer(1024 * sizeof(float));

    manager.initializeWithPattern(send, DataType::Float32, 0, 1024);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CollectiveRunner runner(&comm, stream);
    runner.run(CollectiveOp::AllReduce, send.data(), recv.data(),
               1024, DataType::Float32, ReduceOp::Sum, 0);

    cudaStreamSynchronize(stream);

    // Verify result
    std::vector<float> host(1024);
    recv.copyToHost(host.data(), 1024 * sizeof(float));

    // With single GPU, result should equal input
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(host[i], 1.0f);
    }

    cudaStreamDestroy(stream);
}

TEST_F(CollectiveRunnerTest, BroadcastSingleGpu) {
    NcclCommunicator comm;
    comm.initAll(1);

    MemoryManager manager;
    DeviceBuffer send = manager.allocateSendBuffer(512 * sizeof(float));
    DeviceBuffer recv = manager.allocateRecvBuffer(512 * sizeof(float));

    manager.initializeWithPattern(send, DataType::Float32, 0, 512);
    recv.zero();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CollectiveRunner runner(&comm, stream);
    runner.run(CollectiveOp::Broadcast, send.data(), recv.data(),
               512, DataType::Float32, ReduceOp::Sum, 0);

    cudaStreamSynchronize(stream);

    std::vector<float> host(512);
    recv.copyToHost(host.data(), 512 * sizeof(float));

    for (size_t i = 0; i < 512; ++i) {
        EXPECT_FLOAT_EQ(host[i], 1.0f);
    }

    cudaStreamDestroy(stream);
}

}  // namespace testing
}  // namespace nperf

#else
TEST(CollectiveTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
