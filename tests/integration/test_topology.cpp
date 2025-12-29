/**
 * Integration tests for TopologyDetector.
 *
 * Tests GPU topology detection (requires GPU + NVML).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/topology/detector.h"

namespace nperf {
namespace testing {

class TopologyDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(TopologyDetectorTest, Construction) {
    TopologyDetector detector;
    SUCCEED();
}

TEST_F(TopologyDetectorTest, DetectReturnsValidTopology) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    // Should have at least one GPU
    EXPECT_GE(info.gpus.size(), 1u);
}

TEST_F(TopologyDetectorTest, GpuCountMatchesSystem) {
    int cudaDeviceCount = getCudaGpuCount();

    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    EXPECT_EQ(info.gpus.size(), static_cast<size_t>(cudaDeviceCount));
}

TEST_F(TopologyDetectorTest, GpuNamesPopulated) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    for (const auto& gpu : info.gpus) {
        EXPECT_FALSE(gpu.name.empty());
    }
}

TEST_F(TopologyDetectorTest, PciAddressesPopulated) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    for (const auto& gpu : info.gpus) {
        // PCI address should be in format like "0000:XX:00.0"
        EXPECT_FALSE(gpu.pciBusId.empty());
    }
}

TEST_F(TopologyDetectorTest, P2PMatrixPopulated) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    if (info.gpus.size() >= 2) {
        EXPECT_EQ(info.p2pMatrix.size(), info.gpus.size());
        EXPECT_EQ(info.p2pMatrix[0].size(), info.gpus.size());
    }
}

TEST_F(TopologyDetectorTest, GetNcclVersion) {
    TopologyDetector detector;
    std::string version = detector.getNcclVersion();

    // Should be non-empty and contain version numbers
    EXPECT_FALSE(version.empty());
}

TEST_F(TopologyDetectorTest, GetHostname) {
    TopologyDetector detector;
    std::string hostname = detector.getHostname();

    EXPECT_FALSE(hostname.empty());
}

TEST_F(TopologyDetectorTest, ComputeCapability) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    for (const auto& gpu : info.gpus) {
        // Compute capability should be valid (at least 3.0 for modern GPUs)
        EXPECT_GE(gpu.computeCapabilityMajor, 3);
        EXPECT_GE(gpu.computeCapabilityMinor, 0);
    }
}

TEST_F(TopologyDetectorTest, MemoryInfo) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    for (const auto& gpu : info.gpus) {
        // GPUs should have some memory
        EXPECT_GT(gpu.totalMemoryBytes, 0u);
    }
}

class TopologyMultiGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_FEWER_THAN_N_GPUS(2);
    }
};

TEST_F(TopologyMultiGpuTest, P2PMatrixDiagonal) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    // Diagonal should be "Same" link type
    for (size_t i = 0; i < info.gpus.size(); ++i) {
        EXPECT_EQ(info.p2pMatrix[i][i].linkType, LinkType::Same);
    }
}

TEST_F(TopologyMultiGpuTest, P2PMatrixSymmetry) {
    TopologyDetector detector;
    TopologyInfo info = detector.detect();

    for (size_t i = 0; i < info.gpus.size(); ++i) {
        for (size_t j = i + 1; j < info.gpus.size(); ++j) {
            // Link type should be symmetric
            EXPECT_EQ(info.p2pMatrix[i][j].linkType, info.p2pMatrix[j][i].linkType);
        }
    }
}

}  // namespace testing
}  // namespace nperf

#else
TEST(TopologyDetectorTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
