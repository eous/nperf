/**
 * Integration tests for MemoryManager.
 *
 * Tests GPU memory management and pattern initialization (requires GPU).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/memory.h"
#include <vector>

namespace nperf {
namespace testing {

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(MemoryManagerTest, Construction) {
    MemoryManager manager;
    SUCCEED();
}

TEST_F(MemoryManagerTest, ConstructionWithDevice) {
    MemoryManager manager(0);
    SUCCEED();
}

TEST_F(MemoryManagerTest, AllocateSendBuffer) {
    MemoryManager manager;
    DeviceBuffer buffer = manager.allocateSendBuffer(1024);

    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(MemoryManagerTest, AllocateRecvBuffer) {
    MemoryManager manager;
    DeviceBuffer buffer = manager.allocateRecvBuffer(2048);

    EXPECT_EQ(buffer.size(), 2048u);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(MemoryManagerTest, InitializeWithPatternFloat32) {
    MemoryManager manager;
    DeviceBuffer buffer = manager.allocateSendBuffer(256 * sizeof(float));

    manager.initializeWithPattern(buffer, DataType::Float32, 0, 256);

    // Verify pattern
    std::vector<float> host(256);
    buffer.copyToHost(host.data(), 256 * sizeof(float));

    // Pattern is rank+1 = 1.0 for rank 0
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_FLOAT_EQ(host[i], 1.0f);
    }
}

TEST_F(MemoryManagerTest, InitializeWithPatternFloat64) {
    MemoryManager manager;
    DeviceBuffer buffer = manager.allocateSendBuffer(128 * sizeof(double));

    manager.initializeWithPattern(buffer, DataType::Float64, 3, 128);

    std::vector<double> host(128);
    buffer.copyToHost(host.data(), 128 * sizeof(double));

    // Pattern is rank+1 = 4.0 for rank 3
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_DOUBLE_EQ(host[i], 4.0);
    }
}

TEST_F(MemoryManagerTest, InitializeWithPatternInt32) {
    MemoryManager manager;
    DeviceBuffer buffer = manager.allocateSendBuffer(64 * sizeof(int32_t));

    manager.initializeWithPattern(buffer, DataType::Int32, 7, 64);

    std::vector<int32_t> host(64);
    buffer.copyToHost(host.data(), 64 * sizeof(int32_t));

    // Pattern is rank+1 = 8 for rank 7
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(host[i], 8);
    }
}

TEST_F(MemoryManagerTest, FreeMemory) {
    MemoryManager manager;
    size_t freeBytes = manager.freeMemory();

    EXPECT_GT(freeBytes, 0u);
}

TEST_F(MemoryManagerTest, TotalMemory) {
    MemoryManager manager;
    size_t totalBytes = manager.totalMemory();

    EXPECT_GT(totalBytes, 0u);
    EXPECT_GE(totalBytes, manager.freeMemory());
}

}  // namespace testing
}  // namespace nperf

#else
TEST(MemoryManagerTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
