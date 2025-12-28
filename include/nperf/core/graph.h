#pragma once

#include "nperf/log.h"
#include <cuda_runtime.h>
#include <functional>

namespace nperf {

/// CUDA Graph capture and execution wrapper
/// Captures a sequence of CUDA operations and replays them efficiently
class GraphRunner {
public:
    GraphRunner() = default;
    ~GraphRunner();

    // Non-copyable
    GraphRunner(const GraphRunner&) = delete;
    GraphRunner& operator=(const GraphRunner&) = delete;

    // Movable
    GraphRunner(GraphRunner&& other) noexcept;
    GraphRunner& operator=(GraphRunner&& other) noexcept;

    /// Begin graph capture on stream
    void beginCapture(cudaStream_t stream);

    /// End graph capture
    void endCapture();

    /// Launch the captured graph on stream
    void launch(cudaStream_t stream);

    /// Check if graph is captured and ready
    bool isReady() const { return graphExec_ != nullptr; }

    /// Release resources
    void destroy();

    /// Capture a function (convenience wrapper)
    /// Usage: graph.capture(stream, [&]() { ... cuda operations ... });
    template<typename Func>
    void capture(cudaStream_t stream, Func&& func) {
        beginCapture(stream);
        func();
        endCapture();
    }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    cudaStream_t captureStream_ = nullptr;
    bool capturing_ = false;

    void cleanup();
};

/// RAII helper for graph capture
class ScopedGraphCapture {
public:
    ScopedGraphCapture(GraphRunner& runner, cudaStream_t stream)
        : runner_(runner) {
        runner_.beginCapture(stream);
    }

    ~ScopedGraphCapture() noexcept {
        try {
            runner_.endCapture();
        } catch (const std::exception& e) {
            // Log but don't propagate - destructors must not throw
            logError("CUDA graph capture cleanup failed: " + std::string(e.what()));
        } catch (...) {
            logError("CUDA graph capture cleanup failed with unknown error");
        }
    }

    // Non-copyable, non-movable
    ScopedGraphCapture(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture& operator=(const ScopedGraphCapture&) = delete;
    ScopedGraphCapture(ScopedGraphCapture&&) = delete;
    ScopedGraphCapture& operator=(ScopedGraphCapture&&) = delete;

private:
    GraphRunner& runner_;
};

} // namespace nperf
