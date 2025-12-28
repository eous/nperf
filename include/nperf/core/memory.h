#pragma once

#include "nperf/types.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace nperf {

/// RAII wrapper for device memory allocation
class DeviceBuffer {
public:
    /// Create uninitialized buffer
    DeviceBuffer() = default;

    /// Allocate device memory of specified size
    explicit DeviceBuffer(size_t bytes);

    /// Destructor frees memory
    ~DeviceBuffer();

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    /// Allocate or reallocate buffer
    void allocate(size_t bytes);

    /// Free buffer
    void free();

    /// Get raw pointer
    void* data() { return data_; }
    const void* data() const { return data_; }

    /// Get typed pointer
    template<typename T>
    T* as() { return static_cast<T*>(data_); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data_); }

    /// Get size in bytes
    size_t size() const { return size_; }

    /// Check if allocated
    bool valid() const { return data_ != nullptr; }

    /// Fill buffer with zeros
    void zero();

    /// Fill buffer with a pattern (byte value)
    void fill(unsigned char value);

    /// Copy data from host to device
    void copyFromHost(const void* hostData, size_t bytes);

    /// Copy data from device to host
    void copyToHost(void* hostData, size_t bytes) const;

    /// Async copy from host
    void copyFromHostAsync(const void* hostData, size_t bytes, cudaStream_t stream);

    /// Async copy to host
    void copyToHostAsync(void* hostData, size_t bytes, cudaStream_t stream) const;

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

/// Manager for allocating multiple buffers on a specific GPU
class MemoryManager {
public:
    /// Create manager for specific GPU (or current if -1)
    explicit MemoryManager(int deviceId = -1);

    /// Set device context
    void setDevice();

    /// Allocate send buffer
    DeviceBuffer allocateSendBuffer(size_t bytes);

    /// Allocate receive buffer
    DeviceBuffer allocateRecvBuffer(size_t bytes);

    /// Initialize buffer with test pattern for verification
    /// Pattern: each element = (rank * stride + element_index) mod maxval
    void initializeWithPattern(DeviceBuffer& buffer, DataType dtype,
                               int rank, size_t count);

    /// Get device ID
    int deviceId() const { return deviceId_; }

    /// Get total allocated bytes
    size_t totalAllocated() const { return totalAllocated_; }

    /// Query free memory
    size_t freeMemory() const;

    /// Query total memory
    size_t totalMemory() const;

private:
    int deviceId_;
    size_t totalAllocated_ = 0;
};

/// Pinned (page-locked) host memory for faster transfers
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t bytes);
    ~PinnedBuffer();

    // Non-copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Movable
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void allocate(size_t bytes);
    void free();

    void* data() { return data_; }
    const void* data() const { return data_; }

    template<typename T>
    T* as() { return static_cast<T*>(data_); }

    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace nperf
