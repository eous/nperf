#include "nperf/core/graph.h"
#include <stdexcept>
#include <string>

namespace nperf {

GraphRunner::~GraphRunner() {
    cleanup();
}

GraphRunner::GraphRunner(GraphRunner&& other) noexcept
    : graph_(other.graph_)
    , graphExec_(other.graphExec_)
    , captureStream_(other.captureStream_)
    , capturing_(other.capturing_) {

    other.graph_ = nullptr;
    other.graphExec_ = nullptr;
    other.captureStream_ = nullptr;
    other.capturing_ = false;
}

GraphRunner& GraphRunner::operator=(GraphRunner&& other) noexcept {
    if (this != &other) {
        cleanup();

        graph_ = other.graph_;
        graphExec_ = other.graphExec_;
        captureStream_ = other.captureStream_;
        capturing_ = other.capturing_;

        other.graph_ = nullptr;
        other.graphExec_ = nullptr;
        other.captureStream_ = nullptr;
        other.capturing_ = false;
    }
    return *this;
}

void GraphRunner::cleanup() {
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    captureStream_ = nullptr;
    capturing_ = false;
}

void GraphRunner::beginCapture(cudaStream_t stream) {
    if (capturing_) {
        throw std::runtime_error("GraphRunner: already capturing");
    }

    // Destroy any existing graph
    cleanup();

    captureStream_ = stream;

    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaStreamBeginCapture failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    capturing_ = true;
}

void GraphRunner::endCapture() {
    if (!capturing_) {
        throw std::runtime_error("GraphRunner: not capturing");
    }

    cudaError_t err = cudaStreamEndCapture(captureStream_, &graph_);
    capturing_ = false;

    if (err != cudaSuccess) {
        throw std::runtime_error("cudaStreamEndCapture failed: " +
                                std::string(cudaGetErrorString(err)));
    }

    if (!graph_) {
        throw std::runtime_error("GraphRunner: graph capture produced null graph");
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        throw std::runtime_error("cudaGraphInstantiate failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void GraphRunner::launch(cudaStream_t stream) {
    if (!graphExec_) {
        throw std::runtime_error("GraphRunner: no graph to launch");
    }

    cudaError_t err = cudaGraphLaunch(graphExec_, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGraphLaunch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void GraphRunner::destroy() {
    cleanup();
}

} // namespace nperf
