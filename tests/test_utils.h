#pragma once

/**
 * Common test utilities for nperf test suite.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

namespace nperf {
namespace testing {

// ============================================================================
// GPU Availability Macros
// ============================================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>

/**
 * Check if CUDA is available at runtime.
 */
inline bool hasCudaGpu() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

/**
 * Get the number of available CUDA GPUs.
 */
inline int getCudaGpuCount() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

/**
 * Skip test if no GPU is available.
 * Use at the beginning of GPU-dependent tests.
 */
#define SKIP_IF_NO_GPU() \
    do { \
        if (!::nperf::testing::hasCudaGpu()) { \
            GTEST_SKIP() << "No CUDA GPU available"; \
        } \
    } while (0)

/**
 * Skip test if fewer than N GPUs available.
 */
#define SKIP_IF_FEWER_THAN_N_GPUS(n) \
    do { \
        if (::nperf::testing::getCudaGpuCount() < (n)) { \
            GTEST_SKIP() << "Requires at least " << (n) << " GPUs"; \
        } \
    } while (0)

#else

inline bool hasCudaGpu() { return false; }
inline int getCudaGpuCount() { return 0; }

#define SKIP_IF_NO_GPU() \
    GTEST_SKIP() << "CUDA support not compiled"

#define SKIP_IF_FEWER_THAN_N_GPUS(n) \
    GTEST_SKIP() << "CUDA support not compiled"

#endif

// ============================================================================
// Floating-Point Comparison Helpers
// ============================================================================

/**
 * Check if two floating-point values are approximately equal.
 * Uses relative tolerance for large values, absolute for small values.
 */
template<typename T>
inline bool approxEqual(T a, T b, T relTol = 1e-5, T absTol = 1e-8) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;

    T diff = std::abs(a - b);
    T maxVal = std::max(std::abs(a), std::abs(b));

    return diff <= std::max(relTol * maxVal, absTol);
}

/**
 * EXPECT that two floating-point values are approximately equal.
 */
#define EXPECT_APPROX_EQ(a, b) \
    EXPECT_TRUE(::nperf::testing::approxEqual((a), (b))) \
        << "Expected " << (a) << " to be approximately equal to " << (b)

/**
 * EXPECT that two floating-point values are approximately equal with tolerance.
 */
#define EXPECT_APPROX_EQ_TOL(a, b, rel_tol, abs_tol) \
    EXPECT_TRUE(::nperf::testing::approxEqual((a), (b), (rel_tol), (abs_tol))) \
        << "Expected " << (a) << " to be approximately equal to " << (b) \
        << " (relTol=" << (rel_tol) << ", absTol=" << (abs_tol) << ")"

/**
 * ASSERT that two floating-point values are approximately equal.
 */
#define ASSERT_APPROX_EQ(a, b) \
    ASSERT_TRUE(::nperf::testing::approxEqual((a), (b))) \
        << "Expected " << (a) << " to be approximately equal to " << (b)

// ============================================================================
// Statistical Test Helpers
// ============================================================================

/**
 * Calculate mean of a vector.
 */
inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / static_cast<double>(v.size());
}

/**
 * Calculate standard deviation of a vector.
 */
inline double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double sum = 0.0;
    for (double x : v) {
        double diff = x - m;
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<double>(v.size()));
}

/**
 * Calculate percentile from sorted vector.
 */
inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
    return sorted[idx];
}

// ============================================================================
// Test Data Generators
// ============================================================================

/**
 * Generate a vector of N linearly-spaced doubles in [min, max].
 * Values are deterministic and evenly distributed for reproducible tests.
 */
inline std::vector<double> linearDoubles(size_t n, double minVal = 0.0, double maxVal = 100.0) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n + 1);
        result[i] = minVal + t * (maxVal - minVal);
    }
    return result;
}

/**
 * Generate a vector of N sequential doubles starting from start.
 */
inline std::vector<double> sequentialDoubles(size_t n, double start = 1.0, double step = 1.0) {
    std::vector<double> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// ============================================================================
// String Test Helpers
// ============================================================================

/**
 * Check if a string contains a substring.
 */
inline bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/**
 * Check if a string starts with a prefix.
 */
inline bool startsWith(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

/**
 * Check if a string ends with a suffix.
 */
inline bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace testing
}  // namespace nperf
