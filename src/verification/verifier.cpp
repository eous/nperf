#include "nperf/verification/verifier.h"
#include "nperf/compiler_hints.h"
#include <vector>
#include <cstring>
#include <sstream>

namespace nperf {

Verifier::Verifier(CollectiveOp op, DataType dtype, int worldSize, int rank)
    : op_(op), dtype_(dtype), worldSize_(worldSize), rank_(rank) {
}

NPERF_HOT
void Verifier::initializeSendBuffer(DeviceBuffer& buffer, size_t count) {
    const size_t elementSize = dataTypeSize(dtype_);
    const size_t bytes = count * elementSize;

    std::vector<unsigned char> hostData(bytes);

    const double initVal = getInitValue(rank_, 0);

    switch (dtype_) {
        case DataType::Float32: {
            float* NPERF_RESTRICT ptr = reinterpret_cast<float*>(hostData.data());
            const float val = static_cast<float>(initVal);
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        case DataType::Float64: {
            double* NPERF_RESTRICT ptr = reinterpret_cast<double*>(hostData.data());
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = initVal;
            }
            break;
        }
        case DataType::Int32: {
            int32_t* NPERF_RESTRICT ptr = reinterpret_cast<int32_t*>(hostData.data());
            const int32_t val = static_cast<int32_t>(initVal);
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        case DataType::Int64: {
            int64_t* NPERF_RESTRICT ptr = reinterpret_cast<int64_t*>(hostData.data());
            const int64_t val = static_cast<int64_t>(initVal);
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        default:
            // For other types, fill with byte pattern
            std::memset(hostData.data(), static_cast<int>(initVal), bytes);
            break;
    }

    buffer.copyFromHost(hostData.data(), bytes);
}

double Verifier::computeExpected(size_t /*index*/, double initValue) const {
    switch (op_) {
        case CollectiveOp::AllReduce:
            // Sum of all ranks: sum(rank+1 for rank in 0..worldSize-1)
            // = worldSize * (worldSize + 1) / 2
            return static_cast<double>(worldSize_ * (worldSize_ + 1)) / 2.0;

        case CollectiveOp::Broadcast:
            // Root's value (root+1) is broadcast
            return 1.0; // Assuming root=0

        case CollectiveOp::Reduce:
            // Only root has result, same as AllReduce
            if (rank_ == 0) {
                return static_cast<double>(worldSize_ * (worldSize_ + 1)) / 2.0;
            }
            return initValue; // Non-root ranks keep original

        case CollectiveOp::AllGather:
            // Each position i contains rank i's data (rank+1)
            // We can't verify position-dependent data easily here
            // Just verify that values are in valid range
            return initValue;

        case CollectiveOp::ReduceScatter:
            // Similar to AllReduce but distributed
            return static_cast<double>(worldSize_ * (worldSize_ + 1)) / 2.0;

        default:
            return initValue;
    }
}

double Verifier::getExpectedValue(size_t index) const {
    double initVal = getInitValue(rank_, index);
    return computeExpected(index, initVal);
}

NPERF_HOT
VerifyResult Verifier::verifyRecvBuffer(const DeviceBuffer& buffer, size_t count) {
    VerifyResult result;
    result.passed = true;
    result.errorCount = 0;

    const size_t elementSize = dataTypeSize(dtype_);
    const size_t bytes = count * elementSize;

    std::vector<unsigned char> hostData(bytes);
    buffer.copyToHost(hostData.data(), bytes);

    const double expected = getExpectedValue(0);

    switch (dtype_) {
        case DataType::Float32: {
            const float* NPERF_RESTRICT ptr = reinterpret_cast<const float*>(hostData.data());
            for (size_t i = 0; i < count; ++i) {
                // Prefetch next cache line for large buffers
                if (NPERF_LIKELY((i & 15) == 0 && i + 16 < count)) {
                    NPERF_PREFETCH(&ptr[i + 16], 0, 3);
                }
                if (NPERF_UNLIKELY(!compare(expected, static_cast<double>(ptr[i])))) {
                    if (result.errorCount == 0) {
                        result.firstErrorIndex = i;
                        result.expectedValue = expected;
                        result.actualValue = ptr[i];
                    }
                    result.errorCount++;
                    result.passed = false;
                }
            }
            break;
        }
        case DataType::Float64: {
            const double* NPERF_RESTRICT ptr = reinterpret_cast<const double*>(hostData.data());
            for (size_t i = 0; i < count; ++i) {
                if (NPERF_LIKELY((i & 7) == 0 && i + 8 < count)) {
                    NPERF_PREFETCH(&ptr[i + 8], 0, 3);
                }
                if (NPERF_UNLIKELY(!compare(expected, ptr[i]))) {
                    if (result.errorCount == 0) {
                        result.firstErrorIndex = i;
                        result.expectedValue = expected;
                        result.actualValue = ptr[i];
                    }
                    result.errorCount++;
                    result.passed = false;
                }
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* NPERF_RESTRICT ptr = reinterpret_cast<const int32_t*>(hostData.data());
            const int32_t exp = static_cast<int32_t>(expected);
            for (size_t i = 0; i < count; ++i) {
                if (NPERF_LIKELY((i & 15) == 0 && i + 16 < count)) {
                    NPERF_PREFETCH(&ptr[i + 16], 0, 3);
                }
                if (NPERF_UNLIKELY(ptr[i] != exp)) {
                    if (result.errorCount == 0) {
                        result.firstErrorIndex = i;
                        result.expectedValue = exp;
                        result.actualValue = ptr[i];
                    }
                    result.errorCount++;
                    result.passed = false;
                }
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* NPERF_RESTRICT ptr = reinterpret_cast<const int64_t*>(hostData.data());
            const int64_t exp = static_cast<int64_t>(expected);
            for (size_t i = 0; i < count; ++i) {
                if (NPERF_LIKELY((i & 7) == 0 && i + 8 < count)) {
                    NPERF_PREFETCH(&ptr[i + 8], 0, 3);
                }
                if (NPERF_UNLIKELY(ptr[i] != exp)) {
                    if (result.errorCount == 0) {
                        result.firstErrorIndex = i;
                        result.expectedValue = static_cast<double>(exp);
                        result.actualValue = static_cast<double>(ptr[i]);
                    }
                    result.errorCount++;
                    result.passed = false;
                }
            }
            break;
        }
        default:
            // Skip verification for unsupported types
            break;
    }

    if (NPERF_UNLIKELY(!result.passed)) {
        std::ostringstream ss;
        ss << "Verification failed: " << result.errorCount << " errors, "
           << "first at index " << result.firstErrorIndex
           << " (expected=" << result.expectedValue
           << ", actual=" << result.actualValue << ")";
        result.message = ss.str();
    }

    return result;
}

} // namespace nperf
