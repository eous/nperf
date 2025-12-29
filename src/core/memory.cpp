#include "nperf/core/memory.h"
#include "nperf/compiler_hints.h"
#include <stdexcept>
#include <cstring>

namespace nperf {

// DeviceBuffer implementation

DeviceBuffer::DeviceBuffer(size_t bytes) {
    allocate(bytes);
}

DeviceBuffer::~DeviceBuffer() {
    free();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DeviceBuffer::allocate(size_t bytes) {
    if (data_ && size_ >= bytes) {
        return; // Already have enough
    }

    free();

    if (bytes == 0) return;

    cudaError_t err = cudaMalloc(&data_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    size_ = bytes;
}

void DeviceBuffer::free() {
    if (data_) {
        cudaFree(data_);
        data_ = nullptr;
        size_ = 0;
    }
}

void DeviceBuffer::zero() {
    if (data_ && size_ > 0) {
        cudaError_t err = cudaMemset(data_, 0, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
}

void DeviceBuffer::fill(unsigned char value) {
    if (data_ && size_ > 0) {
        cudaError_t err = cudaMemset(data_, value, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
}

void DeviceBuffer::copyFromHost(const void* hostData, size_t bytes) {
    if (!data_ || bytes > size_) {
        throw std::runtime_error("DeviceBuffer: insufficient buffer size");
    }
    cudaError_t err = cudaMemcpy(data_, hostData, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy H2D failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void DeviceBuffer::copyToHost(void* hostData, size_t bytes) const {
    if (!data_ || bytes > size_) {
        throw std::runtime_error("DeviceBuffer: insufficient buffer size");
    }
    cudaError_t err = cudaMemcpy(hostData, data_, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy D2H failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void DeviceBuffer::copyFromHostAsync(const void* hostData, size_t bytes, cudaStream_t stream) {
    if (!data_ || bytes > size_) {
        throw std::runtime_error("DeviceBuffer: insufficient buffer size");
    }
    cudaError_t err = cudaMemcpyAsync(data_, hostData, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync H2D failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void DeviceBuffer::copyToHostAsync(void* hostData, size_t bytes, cudaStream_t stream) const {
    if (!data_ || bytes > size_) {
        throw std::runtime_error("DeviceBuffer: insufficient buffer size");
    }
    cudaError_t err = cudaMemcpyAsync(hostData, data_, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyAsync D2H failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// MemoryManager implementation

MemoryManager::MemoryManager(int deviceId) : deviceId_(deviceId) {
    if (deviceId_ < 0) {
        cudaError_t err = cudaGetDevice(&deviceId_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get current CUDA device: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
}

void MemoryManager::setDevice() {
    cudaError_t err = cudaSetDevice(deviceId_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device " +
                                std::to_string(deviceId_) + ": " +
                                cudaGetErrorString(err));
    }
}

DeviceBuffer MemoryManager::allocateSendBuffer(size_t bytes) {
    setDevice();
    DeviceBuffer buf(bytes);
    totalAllocated_ += bytes;
    return buf;
}

DeviceBuffer MemoryManager::allocateRecvBuffer(size_t bytes) {
    setDevice();
    DeviceBuffer buf(bytes);
    totalAllocated_ += bytes;
    return buf;
}

NPERF_HOT
void MemoryManager::initializeWithPattern(DeviceBuffer& buffer, DataType dtype,
                                          int rank, size_t count) {
    // For simplicity, we initialize on CPU and copy
    // In production, a CUDA kernel would be more efficient
    const size_t elementSize = dataTypeSize(dtype);
    const size_t bytes = count * elementSize;

    std::vector<unsigned char> hostData(bytes);

    // Simple pattern: value = (rank + 1) for each element
    // This allows verification: after AllReduce sum, each element = sum(ranks) + worldSize
    switch (dtype) {
        case DataType::Float32: {
            float* NPERF_RESTRICT ptr = reinterpret_cast<float*>(hostData.data());
            const float val = static_cast<float>(rank + 1);
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        case DataType::Float64: {
            double* NPERF_RESTRICT ptr = reinterpret_cast<double*>(hostData.data());
            const double val = static_cast<double>(rank + 1);
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        case DataType::Int32: {
            int32_t* NPERF_RESTRICT ptr = reinterpret_cast<int32_t*>(hostData.data());
            const int32_t val = rank + 1;
            NPERF_IVDEP
            for (size_t i = 0; i < count; ++i) {
                ptr[i] = val;
            }
            break;
        }
        default: {
            // For other types, just use rank value as byte pattern
            std::memset(hostData.data(), rank + 1, bytes);
            break;
        }
    }

    buffer.copyFromHost(hostData.data(), bytes);
}

size_t MemoryManager::freeMemory() const {
    size_t free = 0, total = 0;
    int prevDevice;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(deviceId_);
    cudaMemGetInfo(&free, &total);
    cudaSetDevice(prevDevice);
    return free;
}

size_t MemoryManager::totalMemory() const {
    size_t free = 0, total = 0;
    int prevDevice;
    cudaGetDevice(&prevDevice);
    cudaSetDevice(deviceId_);
    cudaMemGetInfo(&free, &total);
    cudaSetDevice(prevDevice);
    return total;
}

// PinnedBuffer implementation

PinnedBuffer::PinnedBuffer(size_t bytes) {
    allocate(bytes);
}

PinnedBuffer::~PinnedBuffer() {
    free();
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        free();
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void PinnedBuffer::allocate(size_t bytes) {
    if (data_ && size_ >= bytes) {
        return;
    }

    free();

    if (bytes == 0) return;

    cudaError_t err = cudaMallocHost(&data_, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed: " +
                                std::string(cudaGetErrorString(err)));
    }
    size_ = bytes;
}

void PinnedBuffer::free() {
    if (data_) {
        cudaFreeHost(data_);
        data_ = nullptr;
        size_ = 0;
    }
}

} // namespace nperf
