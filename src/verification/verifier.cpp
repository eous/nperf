#include "nperf/verification/verifier.h"
#include <vector>
#include <cstring>
#include <sstream>

namespace nperf {

Verifier::Verifier(CollectiveOp op, DataType dtype, int worldSize, int rank)
    : op_(op), dtype_(dtype), worldSize_(worldSize), rank_(rank) {
}

void Verifier::initializeSendBuffer(DeviceBuffer& buffer, size_t count) {
    size_t elementSize = dataTypeSize(dtype_);
    size_t bytes = count * elementSize;

    std::vector<unsigned char> hostData(bytes);

    double initVal = getInitValue(rank_, 0);

    switch (dtype_) {
        case DataType::Float32: {
            float* ptr = reinterpret_cast<float*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                ptr[i] = static_cast<float>(initVal);
            }
            break;
        }
        case DataType::Float64: {
            double* ptr = reinterpret_cast<double*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                ptr[i] = initVal;
            }
            break;
        }
        case DataType::Int32: {
            int32_t* ptr = reinterpret_cast<int32_t*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                ptr[i] = static_cast<int32_t>(initVal);
            }
            break;
        }
        case DataType::Int64: {
            int64_t* ptr = reinterpret_cast<int64_t*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                ptr[i] = static_cast<int64_t>(initVal);
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

VerifyResult Verifier::verifyRecvBuffer(const DeviceBuffer& buffer, size_t count) {
    VerifyResult result;
    result.passed = true;
    result.errorCount = 0;

    size_t elementSize = dataTypeSize(dtype_);
    size_t bytes = count * elementSize;

    std::vector<unsigned char> hostData(bytes);
    buffer.copyToHost(hostData.data(), bytes);

    double expected = getExpectedValue(0);

    switch (dtype_) {
        case DataType::Float32: {
            float* ptr = reinterpret_cast<float*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                if (!compare(expected, static_cast<double>(ptr[i]))) {
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
            double* ptr = reinterpret_cast<double*>(hostData.data());
            for (size_t i = 0; i < count; i++) {
                if (!compare(expected, ptr[i])) {
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
            int32_t* ptr = reinterpret_cast<int32_t*>(hostData.data());
            int32_t exp = static_cast<int32_t>(expected);
            for (size_t i = 0; i < count; i++) {
                if (ptr[i] != exp) {
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
            int64_t* ptr = reinterpret_cast<int64_t*>(hostData.data());
            int64_t exp = static_cast<int64_t>(expected);
            for (size_t i = 0; i < count; i++) {
                if (ptr[i] != exp) {
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

    if (!result.passed) {
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
