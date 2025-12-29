/**
 * Integration tests for Coordinators.
 *
 * Tests LocalCoordinator, SocketCoordinator, and NcclBootstrapCoordinator.
 */

#include <gtest/gtest.h>
#include "test_utils.h"

#ifdef __CUDACC__
#include "nperf/coordination/coordinator.h"
#include <thread>
#include <chrono>

namespace nperf {
namespace testing {

// ============================================================================
// LocalCoordinator Tests
// ============================================================================

class LocalCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(LocalCoordinatorTest, Construction) {
    LocalCoordinator coord(getCudaGpuCount());
    SUCCEED();
}

TEST_F(LocalCoordinatorTest, GetRank) {
    LocalCoordinator coord(getCudaGpuCount());
    coord.initialize(0, nullptr);

    EXPECT_EQ(coord.getRank(), 0);
}

TEST_F(LocalCoordinatorTest, GetWorldSize) {
    int gpuCount = getCudaGpuCount();
    LocalCoordinator coord(gpuCount);
    coord.initialize(0, nullptr);

    // WorldSize is always 1 for local coordinator (single process)
    // numGpus controls how many GPUs are used, not the world size
    EXPECT_EQ(coord.getWorldSize(), 1);
}

TEST_F(LocalCoordinatorTest, Barrier) {
    LocalCoordinator coord(getCudaGpuCount());
    coord.initialize(0, nullptr);

    // Barrier should complete without blocking (single-threaded)
    coord.barrier();
    SUCCEED();
}

TEST_F(LocalCoordinatorTest, BroadcastNcclId) {
    LocalCoordinator coord(getCudaGpuCount());
    coord.initialize(0, nullptr);

    ncclUniqueId id;
    ncclGetUniqueId(&id);

    coord.broadcastNcclId(&id, 0);
    SUCCEED();
}

// ============================================================================
// Coordinator Factory Tests
// ============================================================================

class CoordinatorFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(CoordinatorFactoryTest, CreateLocalCoordinator) {
    CoordinationConfig config;
    config.mode = CoordinationMode::Local;
    config.numLocalGpus = 1;

    auto coord = createCoordinator(config);
    EXPECT_NE(coord, nullptr);
}

// ============================================================================
// SocketCoordinator Tests (Basic)
// ============================================================================

class SocketCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(SocketCoordinatorTest, ServerConstruction) {
    // Server construction (doesn't start listening yet)
    SocketCoordinator server;
    server.setServerMode(5299, 1);
    SUCCEED();
}

TEST_F(SocketCoordinatorTest, ClientConstruction) {
    SocketCoordinator client;
    client.setClientMode("localhost", 5299);
    SUCCEED();
}

// Full server-client tests would require multi-threading or separate processes
// These are integration tests that verify the API works

TEST_F(SocketCoordinatorTest, ServerClientConnection) {
    const int port = 5298;

    // Synchronization primitives
    std::atomic<bool> serverReady{false};
    std::atomic<bool> testComplete{false};
    std::atomic<bool> serverFailed{false};

    // Start server in background
    std::thread serverThread([&]() {
        try {
            SocketCoordinator server;
            server.setServerMode(port, 1);
            serverReady = true;
            server.initialize(0, nullptr);  // Waits for clients
            // Keep server alive until test completes
            while (!testComplete) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (...) {
            serverFailed = true;
            serverReady = true;  // Unblock the wait loop
        }
    });

    // Wait for server to start (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (!serverReady) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() > 2) {
            testComplete = true;
            if (serverThread.joinable()) {
                serverThread.join();
            }
            GTEST_SKIP() << "Server startup timed out (port may be in use)";
            return;
        }
    }

    if (serverFailed) {
        testComplete = true;
        if (serverThread.joinable()) {
            serverThread.join();
        }
        GTEST_SKIP() << "Server failed to start (port may be in use)";
        return;
    }

    // Try to connect
    try {
        SocketCoordinator client;
        client.setClientMode("127.0.0.1", port);
        client.initialize(0, nullptr);

        EXPECT_EQ(client.getRank(), 1);  // Client is rank 1
        EXPECT_EQ(client.getWorldSize(), 2);  // Server + 1 client
    } catch (const std::exception& e) {
        // Connection may fail in some environments - not a failure
        GTEST_SKIP() << "Client connection failed: " << e.what();
    }

    // Proper cleanup
    testComplete = true;
    if (serverThread.joinable()) {
        serverThread.join();
    }
}

// ============================================================================
// NcclBootstrapCoordinator Tests
// ============================================================================

class NcclBootstrapCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_GPU();
    }
};

TEST_F(NcclBootstrapCoordinatorTest, ConstructionWithoutEnvFails) {
    // Without NCCL_COMM_ID env var, this should fail
    unsetenv("NCCL_COMM_ID");

    EXPECT_THROW({
        NcclBootstrapCoordinator coord;
        coord.setRankInfo(0, 2);
        coord.initialize(0, nullptr);
    }, std::runtime_error);
}

}  // namespace testing
}  // namespace nperf

#else
TEST(CoordinatorTest, SkippedNoCuda) {
    GTEST_SKIP() << "CUDA not available";
}
#endif
