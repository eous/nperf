#include "nperf/core/engine.h"
#include "nperf/coordination/socket_coordinator.h"
#include "nperf/coordination/nccl_bootstrap_coordinator.h"
#include "nperf/compiler_hints.h"
#include "nperf/log.h"
#include <stdexcept>
#include <chrono>
#include <cstdlib>

namespace nperf {

BenchmarkEngine::BenchmarkEngine() = default;
BenchmarkEngine::~BenchmarkEngine() {
    finalize();
}

void BenchmarkEngine::configure(const NperfConfig& config) {
    config_ = config;
}

void BenchmarkEngine::initialize(int argc, char** argv) {
    if (initialized_) return;

    logInfo("Initializing benchmark engine...");

    // Set NCCL environment variables based on config
    setNcclEnvVars();

    // Create coordinator
    const char* modeNames[] = {"Local", "MPI", "Socket", "NcclBootstrap"};
    logInfo("Creating coordinator (mode: " +
            std::string(modeNames[static_cast<int>(config_.coordination.mode)]) + ")");
    coordinator_ = Coordinator::create(config_.coordination.mode);

    // Configure NcclBootstrap coordinator if needed
    if (config_.coordination.mode == CoordinationMode::NcclBootstrap) {
        auto* bootstrapCoord = dynamic_cast<NcclBootstrapCoordinator*>(coordinator_.get());
        if (bootstrapCoord) {
            logInfo("NCCL bootstrap mode: rank " + std::to_string(config_.coordination.rank) +
                   ", world size " + std::to_string(config_.coordination.worldSize));
            bootstrapCoord->setRankInfo(config_.coordination.rank,
                                        config_.coordination.worldSize);
        }
    }

    // Configure socket coordinator if needed
    if (config_.coordination.mode == CoordinationMode::Socket) {
        auto* sockCoord = dynamic_cast<SocketCoordinator*>(coordinator_.get());
        if (sockCoord) {
            if (config_.coordination.isServer) {
                logInfo("Socket server mode: port " + std::to_string(config_.coordination.port) +
                       ", expecting " + std::to_string(config_.coordination.expectedClients) + " clients");
                sockCoord->setServerMode(config_.coordination.port,
                                        config_.coordination.expectedClients);
            } else {
                logInfo("Socket client mode: connecting to " + config_.coordination.serverHost +
                       ":" + std::to_string(config_.coordination.port));
                sockCoord->setClientMode(config_.coordination.serverHost,
                                        config_.coordination.port);
            }
        }
    }

    logInfo("Initializing coordinator...");
    coordinator_->initialize(argc, argv);
    logInfo("Rank " + std::to_string(coordinator_->getRank()) + " of " +
           std::to_string(coordinator_->getWorldSize()) + " initialized");

    // Detect topology
    logInfo("Detecting GPU topology...");
    TopologyDetector detector;
    topology_ = detector.detect();
    logInfo("Found " + std::to_string(topology_.gpus.size()) + " GPUs" +
           (topology_.hasNVSwitch ? " (NVSwitch detected)" : ""));

    // Select CUDA device
    int deviceId = config_.benchmark.cudaDevice;
    if (deviceId < 0) {
        // Auto-select based on rank
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device count: " +
                                    std::string(cudaGetErrorString(err)));
        }
        if (deviceCount <= 0) {
            throw std::runtime_error("No CUDA devices available");
        }
        deviceId = coordinator_->getRank() % deviceCount;
        logInfo("Auto-selected GPU " + std::to_string(deviceId) + " for rank " +
               std::to_string(coordinator_->getRank()));
    } else {
        logInfo("Using configured GPU " + std::to_string(deviceId));
    }

    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device " + std::to_string(deviceId) +
                                ": " + std::string(cudaGetErrorString(err)));
    }

    // Log GPU info
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, deviceId) == cudaSuccess) {
        logInfo("GPU " + std::to_string(deviceId) + ": " + std::string(props.name));
    }

    // Create stream
    logDebug("Creating CUDA stream...");
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " +
                                std::string(cudaGetErrorString(err)));
    }

    // Setup memory manager
    logDebug("Initializing memory manager...");
    memoryManager_ = std::make_unique<MemoryManager>(deviceId);

    initialized_ = true;
    logInfo("Engine initialization complete");
}

void BenchmarkEngine::setupNccl() {
    logInfo("Setting up NCCL communicator...");

    // Get or generate NCCL unique ID
    ncclUniqueId ncclId;
    if (coordinator_->isRoot()) {
        logDebug("Root generating NCCL unique ID...");
        ncclId = NcclCommunicator::getUniqueId();
    }

    // Broadcast ID to all ranks
    logDebug("Broadcasting NCCL ID to all ranks...");
    coordinator_->broadcastNcclId(&ncclId, 0);

    // Initialize NCCL communicator
    logDebug("Initializing NCCL communicator for " +
            std::to_string(coordinator_->getWorldSize()) + " ranks...");
    ncclComm_.initRank(coordinator_->getWorldSize(), ncclId, coordinator_->getRank());

    // Create collective runner
    logDebug("Creating collective runner...");
    runner_ = std::make_unique<CollectiveRunner>(ncclComm_.handle(), stream_);

    logInfo("NCCL setup complete");
}

void BenchmarkEngine::allocateBuffers(size_t maxBytes) {
    // For AllGather, receive buffer needs worldSize * sendSize
    size_t recvMultiplier = 1;
    if (config_.benchmark.operation == CollectiveOp::AllGather) {
        recvMultiplier = coordinator_->getWorldSize();
    }

    size_t sendSize = maxBytes;
    size_t recvSize = maxBytes * recvMultiplier;
    logInfo("Allocating buffers: send=" + formatSize(sendSize) +
           ", recv=" + formatSize(recvSize));

    sendBuffer_ = memoryManager_->allocateSendBuffer(sendSize);
    recvBuffer_ = memoryManager_->allocateRecvBuffer(recvSize);

    // Zero buffers
    logDebug("Zeroing buffers...");
    sendBuffer_.zero();
    recvBuffer_.zero();
}

void BenchmarkEngine::captureGraph(size_t count) {
    if (!config_.benchmark.useCudaGraph || !graphRunner_) {
        return;
    }

    // Capture the collective operation into a graph
    graphRunner_->capture(stream_, [&]() {
        runner_->run(
            config_.benchmark.operation,
            sendBuffer_.data(),
            recvBuffer_.data(),
            count,
            config_.benchmark.dataType,
            config_.benchmark.reduceOp,
            config_.benchmark.rootRank
        );
    });
}

void BenchmarkEngine::runWarmup(size_t bytes) {
    size_t count = bytes / dataTypeSize(config_.benchmark.dataType);

    for (int i = 0; i < config_.benchmark.warmupIterations; i++) {
        runner_->run(
            config_.benchmark.operation,
            sendBuffer_.data(),
            recvBuffer_.data(),
            count,
            config_.benchmark.dataType,
            config_.benchmark.reduceOp,
            config_.benchmark.rootRank
        );
    }

    cudaStreamSynchronize(stream_);
}

NPERF_HOT
SizeResult BenchmarkEngine::runSize(size_t bytes) {
    SizeResult result;
    result.messageBytes = bytes;

    const size_t elementSize = dataTypeSize(config_.benchmark.dataType);
    const size_t count = bytes / elementSize;
    result.elementCount = count;

    // Initialize buffer for verification if needed
    if (verifier_ && config_.benchmark.verifyMode != VerifyMode::None) {
        verifier_->initializeSendBuffer(sendBuffer_, count);
    }

    // Capture graph if using graph mode
    if (config_.benchmark.useCudaGraph) {
        if (!graphRunner_) {
            graphRunner_ = std::make_unique<GraphRunner>();
        }
        captureGraph(count);
    }

    CudaTimer timer(stream_);
    std::vector<double> latencies;

    result.verified = true;
    result.verifyErrors = 0;

    // Cache frequently accessed config values
    const bool useGraph = config_.benchmark.useCudaGraph;
    const bool doVerify = verifier_ && config_.benchmark.verifyMode == VerifyMode::PerIteration;
    const CollectiveOp operation = config_.benchmark.operation;
    const DataType dataType = config_.benchmark.dataType;
    const ReduceOp reduceOp = config_.benchmark.reduceOp;
    const int rootRank = config_.benchmark.rootRank;
    void* NPERF_RESTRICT sendData = sendBuffer_.data();
    void* NPERF_RESTRICT recvData = recvBuffer_.data();

    if (config_.benchmark.useTimeBased) {
        // Time-based mode: run for specified duration
        const auto testDurationUs = static_cast<int64_t>(config_.benchmark.testDurationSeconds * SECONDS_TO_US);
        const auto omitDurationUs = static_cast<int64_t>(config_.benchmark.omitSeconds * SECONDS_TO_US);

        auto startTime = std::chrono::high_resolution_clock::now();
        int64_t elapsedUs = 0;

        while (elapsedUs < testDurationUs) {
            timer.start();

            if (NPERF_LIKELY(useGraph && graphRunner_->isReady())) {
                graphRunner_->launch(stream_);
            } else {
                runner_->run(operation, sendData, recvData, count,
                            dataType, reduceOp, rootRank);
            }

            timer.stop();
            const double latencyUs = timer.getElapsedUs();

            // Only record latencies after omit period
            if (NPERF_LIKELY(elapsedUs >= omitDurationUs)) {
                latencies.push_back(latencyUs);
            }

            // Per-iteration verification (cold path)
            if (NPERF_UNLIKELY(doVerify)) {
                cudaStreamSynchronize(stream_);
                auto verifyResult = verifier_->verifyRecvBuffer(recvBuffer_, count);
                if (NPERF_UNLIKELY(!verifyResult.passed)) {
                    result.verified = false;
                    result.verifyErrors += verifyResult.errorCount;
                }
                verifier_->initializeSendBuffer(sendBuffer_, count);
            }

            auto now = std::chrono::high_resolution_clock::now();
            elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(now - startTime).count();
        }
    } else {
        // Iteration-based mode: run for specified number of iterations
        const int iterations = config_.benchmark.iterations;
        latencies.reserve(iterations);

        for (int iter = 0; iter < iterations; ++iter) {
            timer.start();

            if (NPERF_LIKELY(useGraph && graphRunner_->isReady())) {
                graphRunner_->launch(stream_);
            } else {
                runner_->run(operation, sendData, recvData, count,
                            dataType, reduceOp, rootRank);
            }

            timer.stop();
            const double latencyUs = timer.getElapsedUs();
            latencies.push_back(latencyUs);

            // Per-iteration verification (cold path)
            if (NPERF_UNLIKELY(doVerify)) {
                cudaStreamSynchronize(stream_);
                auto verifyResult = verifier_->verifyRecvBuffer(recvBuffer_, count);
                if (NPERF_UNLIKELY(!verifyResult.passed)) {
                    result.verified = false;
                    result.verifyErrors += verifyResult.errorCount;
                }
                verifier_->initializeSendBuffer(sendBuffer_, count);
            }
        }
    }

    // Compute timing statistics
    if (NPERF_UNLIKELY(latencies.empty())) {
        // No valid samples (e.g., omit period was longer than test duration)
        result.timing = TimingStats{};
        result.bandwidth = BandwidthMetrics{};
        result.iterations = 0;
    } else {
        result.timing = TimingStats::compute(latencies);
        result.bandwidth = computeBandwidth(
            bytes,
            result.timing.avgUs,
            config_.benchmark.operation,
            coordinator_->getWorldSize()
        );
        result.iterations = static_cast<int>(latencies.size());
    }

    return result;
}

BenchmarkResults BenchmarkEngine::run() {
    if (!initialized_) {
        throw std::runtime_error("Engine not initialized");
    }

    BenchmarkResults results;
    results.config = config_.benchmark;
    results.topology = topology_;
    results.rank = coordinator_->getRank();
    results.worldSize = coordinator_->getWorldSize();
    results.startTime = std::chrono::system_clock::now();

    // Setup NCCL
    setupNccl();

    // Get message sizes to test
    auto sizes = config_.benchmark.getMessageSizes();
    size_t maxBytes = sizes.empty() ? config_.benchmark.minBytes : sizes.back();

    logInfo("Benchmark: " + std::to_string(sizes.size()) + " sizes from " +
           formatSize(sizes.empty() ? 0 : sizes.front()) + " to " + formatSize(maxBytes));

    // Allocate buffers for largest size
    allocateBuffers(maxBytes);

    // Setup verifier if needed
    if (config_.benchmark.verifyMode != VerifyMode::None) {
        logInfo("Verification enabled (tolerance: " +
               std::to_string(config_.benchmark.verifyTolerance) + ")");
        verifier_ = std::make_unique<Verifier>(
            config_.benchmark.operation,
            config_.benchmark.dataType,
            coordinator_->getWorldSize(),
            coordinator_->getRank()
        );
        verifier_->setTolerance(config_.benchmark.verifyTolerance);
    }

    // Setup metrics calculator
    metrics_.setOperation(config_.benchmark.operation);
    metrics_.setWorldSize(coordinator_->getWorldSize());

    // Synchronize before starting
    logDebug("Barrier before benchmark start...");
    coordinator_->barrier();

    // Run warmup for first size
    if (!sizes.empty() && config_.benchmark.warmupIterations > 0) {
        logInfo("Running " + std::to_string(config_.benchmark.warmupIterations) +
               " warmup iterations...");
        runWarmup(sizes[0]);
    }

    logInfo("Starting benchmark...");

    // Run benchmark for each size
    for (size_t sizeIdx = 0; sizeIdx < sizes.size(); sizeIdx++) {
        size_t bytes = sizes[sizeIdx];

        logInfo("Size " + std::to_string(sizeIdx + 1) + "/" +
               std::to_string(sizes.size()) + ": " + formatSize(bytes));

        auto sizeResult = runSize(bytes);
        results.sizeResults.push_back(sizeResult);

        logInfo("  -> " + std::to_string(sizeResult.timing.avgUs) + " us, " +
               std::to_string(sizeResult.bandwidth.busGBps) + " GB/s" +
               (sizeResult.verified ? "" : " [VERIFY FAILED]"));

        // Report progress if callback set
        if (progressCallback_) {
            IntervalReport report;
            report.bytesTransferred = bytes * sizeResult.iterations;
            report.operationsCompleted = sizeResult.iterations;
            report.currentBandwidthGBps = sizeResult.bandwidth.busGBps;
            report.currentLatencyUs = sizeResult.timing.avgUs;
            report.currentSizeIndex = sizeIdx;
            report.totalSizes = sizes.size();
            report.currentMessageBytes = bytes;
            report.totalIterations = sizeResult.iterations;
            report.overallProgress = static_cast<double>(sizeIdx + 1) / sizes.size();
            progressCallback_(report);
        }
    }

    results.endTime = std::chrono::system_clock::now();
    results.computeSummary();

    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        results.endTime - results.startTime).count();
    logInfo("Benchmark complete in " + std::to_string(durationMs) + " ms");

    return results;
}

TopologyInfo BenchmarkEngine::runTopologyOnly() {
    if (!initialized_) {
        throw std::runtime_error("Engine not initialized");
    }

    TopologyDetector detector;
    return detector.detect();
}

void BenchmarkEngine::setProgressCallback(ProgressCallback callback) {
    progressCallback_ = std::move(callback);
}

int BenchmarkEngine::rank() const {
    return coordinator_ ? coordinator_->getRank() : 0;
}

int BenchmarkEngine::worldSize() const {
    return coordinator_ ? coordinator_->getWorldSize() : 1;
}

void BenchmarkEngine::finalize() {
    if (!initialized_) return;

    // Destroy verifier
    verifier_.reset();

    // Destroy graph
    graphRunner_.reset();

    // Destroy collective runner
    runner_.reset();

    // Destroy NCCL communicator
    ncclComm_.destroy();

    // Release device buffers explicitly before destroying memory manager
    sendBuffer_.free();
    recvBuffer_.free();

    // Destroy memory manager
    memoryManager_.reset();

    // Destroy stream
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    // Finalize coordinator
    if (coordinator_) {
        coordinator_->finalize();
        coordinator_.reset();
    }

    initialized_ = false;
}

void BenchmarkEngine::setNcclEnvVars() {
    logDebug("Configuring NCCL environment...");

    // Set NCCL debug if requested
    if (config_.output.debug) {
        setenv("NCCL_DEBUG", "INFO", 0);
        setenv("NCCL_DEBUG_SUBSYS", "ALL", 0);
        logDebug("NCCL_DEBUG=INFO enabled");
    }

    // Set algorithm if specified
    if (config_.benchmark.algorithm != Algorithm::Auto) {
        const char* algo = algorithmName(config_.benchmark.algorithm);
        setenv("NCCL_ALGO", algo, 0);
        logInfo("NCCL algorithm: " + std::string(algo));
    } else {
        logDebug("NCCL algorithm: Auto");
    }

    // Set protocol if specified
    if (config_.benchmark.protocol != Protocol::Auto) {
        const char* proto = protocolName(config_.benchmark.protocol);
        setenv("NCCL_PROTO", proto, 0);
        logInfo("NCCL protocol: " + std::string(proto));
    } else {
        logDebug("NCCL protocol: Auto");
    }
}

} // namespace nperf
