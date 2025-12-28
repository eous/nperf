#pragma once

#include "nperf/types.h"
#include "nperf/core/memory.h"
#include <string>
#include <cmath>

namespace nperf {

/// Result of verification check
struct VerifyResult {
    bool passed = true;
    int errorCount = 0;
    size_t firstErrorIndex = 0;
    double expectedValue = 0.0;
    double actualValue = 0.0;
    std::string message;
};

/// Verifies correctness of collective operations
class Verifier {
public:
    /// Create verifier for specific operation
    Verifier(CollectiveOp op, DataType dtype, int worldSize, int rank);

    /// Set tolerance for floating point comparisons
    void setTolerance(double tol) { tolerance_ = tol; }

    /// Initialize send buffer with verification pattern
    void initializeSendBuffer(DeviceBuffer& buffer, size_t count);

    /// Verify receive buffer after collective operation
    VerifyResult verifyRecvBuffer(const DeviceBuffer& buffer, size_t count);

    /// Get expected value for element after operation
    double getExpectedValue(size_t index) const;

private:
    CollectiveOp op_;
    DataType dtype_;
    int worldSize_;
    int rank_;
    double tolerance_ = 1e-5;

    /// Compute expected value based on operation type
    double computeExpected(size_t index, double initValue) const;

    /// Compare with tolerance
    bool compare(double expected, double actual) const {
        if (std::abs(expected) < 1e-10) {
            return std::abs(actual) < tolerance_;
        }
        return std::abs(expected - actual) / std::abs(expected) < tolerance_;
    }
};

/// Helper to generate verification pattern
inline double getInitValue(int rank, [[maybe_unused]] size_t index) {
    // Each rank initializes to (rank + 1)
    // This makes verification easy:
    // - AllReduce sum: expected = sum(1..worldSize) = worldSize*(worldSize+1)/2
    // - Broadcast: expected = root+1
    // Note: index parameter reserved for future position-dependent patterns
    return static_cast<double>(rank + 1);
}

} // namespace nperf
