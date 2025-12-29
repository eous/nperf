/**
 * Shared main entry point for nperf test suites.
 *
 * This file provides the main() function for both unit and integration tests.
 * For integration tests, it checks for GPU availability before running.
 */

#include <gtest/gtest.h>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace {

/**
 * Print test environment information.
 */
void printEnvironmentInfo() {
    std::cout << "========================================\n";
    std::cout << "       nperf Test Suite\n";
    std::cout << "========================================\n";

#ifdef __CUDACC__
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess && deviceCount > 0) {
        std::cout << "CUDA Devices: " << deviceCount << "\n";
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  [" << i << "] " << prop.name
                      << " (CC " << prop.major << "." << prop.minor << ")\n";
        }
    } else {
        std::cout << "CUDA: Not available\n";
    }
#else
    std::cout << "CUDA: Not compiled with CUDA support\n";
#endif

    std::cout << "========================================\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Print environment info
    printEnvironmentInfo();

    // Run tests
    return RUN_ALL_TESTS();
}
