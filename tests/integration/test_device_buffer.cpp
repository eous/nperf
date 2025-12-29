/**
 * Integration tests for DeviceBuffer and PinnedBuffer.
 *
 * Tests GPU memory allocation and data transfer (requires GPU).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/memory.h"
#include <vector>
#include <cstring>

namespace nperf {
namespace testing {

class DeviceBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(DeviceBufferTest, ZeroSizeAllocation) {
    DeviceBuffer buffer(0);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST_F(DeviceBufferTest, SmallAllocation) {
    DeviceBuffer buffer(1024);
    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(DeviceBufferTest, LargeAllocation) {
    size_t size = 256 * 1024 * 1024;  // 256 MB
    DeviceBuffer buffer(size);
    EXPECT_EQ(buffer.size(), size);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(DeviceBufferTest, MoveConstruction) {
    DeviceBuffer buffer1(1024);
    void* ptr = buffer1.data();

    DeviceBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.data(), ptr);
    EXPECT_EQ(buffer2.size(), 1024u);
    EXPECT_EQ(buffer1.data(), nullptr);
    EXPECT_EQ(buffer1.size(), 0u);
}

TEST_F(DeviceBufferTest, MoveAssignment) {
    DeviceBuffer buffer1(1024);
    DeviceBuffer buffer2(2048);
    void* ptr = buffer1.data();

    buffer2 = std::move(buffer1);

    EXPECT_EQ(buffer2.data(), ptr);
    EXPECT_EQ(buffer2.size(), 1024u);
}

TEST_F(DeviceBufferTest, CopyRoundTrip) {
    const size_t size = 1024;
    std::vector<unsigned char> hostSrc(size, 0x42);
    std::vector<unsigned char> hostDst(size, 0x00);

    DeviceBuffer buffer(size);
    buffer.copyFromHost(hostSrc.data(), size);
    buffer.copyToHost(hostDst.data(), size);

    EXPECT_EQ(hostSrc, hostDst);
}

TEST_F(DeviceBufferTest, Fill) {
    const size_t size = 1024;
    std::vector<unsigned char> host(size);

    DeviceBuffer buffer(size);
    buffer.fill(0xAB);
    buffer.copyToHost(host.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host[i], 0xAB);
    }
}

TEST_F(DeviceBufferTest, Zero) {
    const size_t size = 1024;
    std::vector<unsigned char> host(size, 0xFF);

    DeviceBuffer buffer(size);
    buffer.fill(0xFF);
    buffer.zero();
    buffer.copyToHost(host.data(), size);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(host[i], 0x00);
    }
}

class PinnedBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(PinnedBufferTest, ZeroSizeAllocation) {
    PinnedBuffer buffer(0);
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_EQ(buffer.data(), nullptr);
}

TEST_F(PinnedBufferTest, SmallAllocation) {
    PinnedBuffer buffer(1024);
    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(PinnedBufferTest, MoveConstruction) {
    PinnedBuffer buffer1(1024);
    void* ptr = buffer1.data();

    PinnedBuffer buffer2(std::move(buffer1));

    EXPECT_EQ(buffer2.data(), ptr);
    EXPECT_EQ(buffer1.data(), nullptr);
}

TEST_F(PinnedBufferTest, DirectAccess) {
    PinnedBuffer buffer(1024);

    // Pinned memory can be directly accessed
    std::memset(buffer.data(), 0x55, buffer.size());

    unsigned char* ptr = static_cast<unsigned char*>(buffer.data());
    for (size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(ptr[i], 0x55);
    }
}

}  // namespace testing
}  // namespace nperf

#else
TEST(DeviceBufferTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
