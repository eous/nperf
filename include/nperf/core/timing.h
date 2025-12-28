#pragma once

#include <cuda_runtime.h>
#include <chrono>

namespace nperf {

/// RAII wrapper for CUDA event-based timing
class CudaTimer {
public:
    /// Create timer for a specific stream
    explicit CudaTimer(cudaStream_t stream = nullptr);

    /// Destructor cleans up CUDA events
    ~CudaTimer();

    // Non-copyable
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    // Movable
    CudaTimer(CudaTimer&& other) noexcept;
    CudaTimer& operator=(CudaTimer&& other) noexcept;

    /// Record start event
    void start();

    /// Record stop event
    void stop();

    /// Synchronize and get elapsed time in milliseconds
    /// Must call stop() first
    float getElapsedMs();

    /// Synchronize and get elapsed time in microseconds
    float getElapsedUs() { return getElapsedMs() * 1000.0f; }

    /// Check if timer is currently running (start called but not stop)
    bool isRunning() const { return running_; }

    /// Get the stream this timer is associated with
    cudaStream_t stream() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t stopEvent_ = nullptr;
    bool running_ = false;
    bool valid_ = false;

    void cleanup();
};

/// Helper to time a block of code using RAII
class ScopedTimer {
public:
    explicit ScopedTimer(CudaTimer& timer) : timer_(timer) {
        timer_.start();
    }

    ~ScopedTimer() {
        timer_.stop();
    }

private:
    CudaTimer& timer_;
};

/// CPU-side high-resolution timer for non-CUDA timing
class CpuTimer {
public:
    void start();
    void stop();
    double getElapsedMs() const;
    double getElapsedUs() const { return getElapsedMs() * 1000.0; }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

} // namespace nperf
