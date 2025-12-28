#include "nperf/core/timing.h"
#include <stdexcept>
#include <chrono>

namespace nperf {

CudaTimer::CudaTimer(cudaStream_t stream)
    : stream_(stream), running_(false), valid_(false) {

    cudaError_t err;

    err = cudaEventCreate(&startEvent_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA start event: " +
                                std::string(cudaGetErrorString(err)));
    }

    err = cudaEventCreate(&stopEvent_);
    if (err != cudaSuccess) {
        cudaEventDestroy(startEvent_);
        throw std::runtime_error("Failed to create CUDA stop event: " +
                                std::string(cudaGetErrorString(err)));
    }

    valid_ = true;
}

CudaTimer::~CudaTimer() {
    cleanup();
}

CudaTimer::CudaTimer(CudaTimer&& other) noexcept
    : stream_(other.stream_)
    , startEvent_(other.startEvent_)
    , stopEvent_(other.stopEvent_)
    , running_(other.running_)
    , valid_(other.valid_) {

    other.startEvent_ = nullptr;
    other.stopEvent_ = nullptr;
    other.valid_ = false;
    other.running_ = false;
}

CudaTimer& CudaTimer::operator=(CudaTimer&& other) noexcept {
    if (this != &other) {
        cleanup();

        stream_ = other.stream_;
        startEvent_ = other.startEvent_;
        stopEvent_ = other.stopEvent_;
        running_ = other.running_;
        valid_ = other.valid_;

        other.startEvent_ = nullptr;
        other.stopEvent_ = nullptr;
        other.valid_ = false;
        other.running_ = false;
    }
    return *this;
}

void CudaTimer::cleanup() {
    if (startEvent_) {
        cudaEventDestroy(startEvent_);
        startEvent_ = nullptr;
    }
    if (stopEvent_) {
        cudaEventDestroy(stopEvent_);
        stopEvent_ = nullptr;
    }
    valid_ = false;
    running_ = false;
}

void CudaTimer::start() {
    if (!valid_) {
        throw std::runtime_error("CudaTimer: timer not valid");
    }

    cudaError_t err = cudaEventRecord(startEvent_, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to record start event: " +
                                std::string(cudaGetErrorString(err)));
    }
    running_ = true;
}

void CudaTimer::stop() {
    if (!valid_) {
        throw std::runtime_error("CudaTimer: timer not valid");
    }
    if (!running_) {
        throw std::runtime_error("CudaTimer: stop() called without start()");
    }

    cudaError_t err = cudaEventRecord(stopEvent_, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to record stop event: " +
                                std::string(cudaGetErrorString(err)));
    }
    running_ = false;
}

float CudaTimer::getElapsedMs() {
    if (!valid_) {
        throw std::runtime_error("CudaTimer: timer not valid");
    }
    if (running_) {
        throw std::runtime_error("CudaTimer: getElapsedMs() called while still running");
    }

    // Synchronize on stop event
    cudaError_t err = cudaEventSynchronize(stopEvent_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize stop event: " +
                                std::string(cudaGetErrorString(err)));
    }

    float elapsedMs = 0.0f;
    err = cudaEventElapsedTime(&elapsedMs, startEvent_, stopEvent_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get elapsed time: " +
                                std::string(cudaGetErrorString(err)));
    }

    return elapsedMs;
}

// CpuTimer implementation
void CpuTimer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop() {
    end_ = std::chrono::high_resolution_clock::now();
}

double CpuTimer::getElapsedMs() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
    return duration.count() / 1000.0;
}

} // namespace nperf
