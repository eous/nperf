/**
 * Integration tests for nperf/core/timing.h
 *
 * Tests CudaTimer and timing accuracy (requires GPU).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/timing.h"
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

namespace nperf {
namespace testing {

class CudaTimerTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(CudaTimerTest, Construction) {
    CudaTimer timer;
    SUCCEED();
}

TEST_F(CudaTimerTest, StartStop) {
    CudaTimer timer;
    timer.start();
    timer.stop();
    float elapsed = timer.getElapsedMs();
    EXPECT_GE(elapsed, 0.0f);
}

TEST_F(CudaTimerTest, TimingAccuracy) {
    CudaTimer timer;
    timer.start();

    // Do something that takes measurable time
    cudaDeviceSynchronize();

    timer.stop();
    float elapsed = timer.getElapsedMs();

    // Should be non-zero but not huge
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_LT(elapsed, 1000.0f);  // Less than 1 second
}

TEST_F(CudaTimerTest, Reset) {
    CudaTimer timer;
    timer.start();
    timer.stop();
    float first = timer.getElapsedMs();

    timer.reset();
    timer.start();
    timer.stop();
    float second = timer.getElapsedMs();

    // Both should be valid
    EXPECT_GE(first, 0.0f);
    EXPECT_GE(second, 0.0f);
}

class CpuTimerTest : public ::testing::Test {};

TEST_F(CpuTimerTest, Construction) {
    CpuTimer timer;
    SUCCEED();
}

TEST_F(CpuTimerTest, TimingAccuracy) {
    CpuTimer timer;
    timer.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    timer.stop();
    double elapsed = timer.getElapsedUs();

    // Should be approximately 10ms = 10000us
    // Use wide tolerance for CI environments under heavy load
    EXPECT_GT(elapsed, 1000.0);    // At least 1ms (loose lower bound)
    EXPECT_LT(elapsed, 100000.0);  // Less than 100ms (loose upper bound)
}

}  // namespace testing
}  // namespace nperf

#else
// No CUDA - provide empty test
TEST(CudaTimerTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
