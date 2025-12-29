/**
 * Integration tests for CUDA Graph capture.
 *
 * Tests GraphRunner (requires GPU).
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/core/graph.h"
#include <cuda_runtime.h>

namespace nperf {
namespace testing {

class GraphRunnerTest : public ::testing::Test {
protected:
    cudaStream_t stream_ = nullptr;
    bool streamCreated_ = false;

    void SetUp() override {
        SKIP_IF_NO_GPU();
        cudaStreamCreate(&stream_);
        streamCreated_ = true;
    }

    void TearDown() override {
        if (streamCreated_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
            streamCreated_ = false;
        }
    }
};

TEST_F(GraphRunnerTest, Construction) {
    GraphRunner runner;
    EXPECT_FALSE(runner.isReady());
}

TEST_F(GraphRunnerTest, BeginEndCapture) {
    GraphRunner runner;

    runner.beginCapture(stream_);

    // Capture some work (empty is fine for testing)
    cudaStreamSynchronize(stream_);

    runner.endCapture();

    EXPECT_TRUE(runner.isReady());
}

TEST_F(GraphRunnerTest, Launch) {
    GraphRunner runner;

    runner.beginCapture(stream_);
    cudaStreamSynchronize(stream_);
    runner.endCapture();

    ASSERT_TRUE(runner.isReady());

    // Should not throw
    runner.launch(stream_);
    cudaStreamSynchronize(stream_);
}

TEST_F(GraphRunnerTest, MultipleLaunches) {
    GraphRunner runner;

    runner.beginCapture(stream_);
    cudaStreamSynchronize(stream_);
    runner.endCapture();

    // Launch multiple times
    for (int i = 0; i < 10; ++i) {
        runner.launch(stream_);
    }
    cudaStreamSynchronize(stream_);
}

TEST_F(GraphRunnerTest, MoveConstruction) {
    GraphRunner runner1;

    runner1.beginCapture(stream_);
    cudaStreamSynchronize(stream_);
    runner1.endCapture();

    GraphRunner runner2(std::move(runner1));

    EXPECT_TRUE(runner2.isReady());
    EXPECT_FALSE(runner1.isReady());
}

TEST_F(GraphRunnerTest, Recapture) {
    GraphRunner runner;

    // First capture
    runner.beginCapture(stream_);
    runner.endCapture();
    EXPECT_TRUE(runner.isReady());

    // Second capture (should replace)
    runner.beginCapture(stream_);
    runner.endCapture();
    EXPECT_TRUE(runner.isReady());
}

class ScopedGraphCaptureTest : public ::testing::Test {
protected:
    cudaStream_t stream_ = nullptr;
    bool streamCreated_ = false;

    void SetUp() override {
        SKIP_IF_NO_GPU();
        cudaStreamCreate(&stream_);
        streamCreated_ = true;
    }

    void TearDown() override {
        if (streamCreated_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
            streamCreated_ = false;
        }
    }
};

TEST_F(ScopedGraphCaptureTest, RAIIBehavior) {
    GraphRunner runner;

    {
        ScopedGraphCapture capture(runner, stream_);
        // Work would be captured here
    }

    // After scope, capture should be complete
    EXPECT_TRUE(runner.isReady());
}

}  // namespace testing
}  // namespace nperf

#else
TEST(GraphRunnerTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
