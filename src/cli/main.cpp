#include "nperf/cli/parser.h"
#include "nperf/core/engine.h"
#include "nperf/output/formatter.h"
#include "nperf/output/topo_visualizer.h"
#include "nperf/version.h"
#include "nperf/log.h"
#include <iostream>
#include <fstream>
#include <csignal>
#include <memory>

using namespace nperf;

// Global engine pointer for signal handling
static BenchmarkEngine* g_engine = nullptr;

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cerr << "\nInterrupted, cleaning up..." << std::endl;
        if (g_engine) {
            g_engine->finalize();
        }
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // Parse arguments
    ArgParser parser;
    if (!parser.parse(argc, argv)) {
        if (parser.helpRequested()) {
            printUsage();
            return 0;
        }
        if (parser.versionRequested()) {
            printVersion();
            return 0;
        }
        std::cerr << "Error: " << parser.errorMessage() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    }

    const auto& config = parser.config();

    // Setup log level based on config
    if (config.output.debug) {
        setLogLevel(LogLevel::Debug);
    } else if (config.output.verbose) {
        setLogLevel(LogLevel::Info);
    } else {
        setLogLevel(LogLevel::Warning);
    }

    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        BenchmarkEngine engine;
        g_engine = &engine;

        engine.configure(config);
        engine.initialize(argc, argv);

        // Select output stream
        std::ostream* outStream = &std::cout;
        std::ofstream fileStream;
        if (!config.output.outputFile.empty()) {
            fileStream.open(config.output.outputFile);
            if (!fileStream) {
                std::cerr << "Error: Cannot open output file: "
                          << config.output.outputFile << std::endl;
                return 1;
            }
            outStream = &fileStream;
        }

        // Create formatter
        auto formatter = Formatter::create(config.output.format);

        // Topology-only mode
        if (config.output.topologyOnly) {
            auto topology = engine.runTopologyOnly();

            if (config.output.topoFormat == TopoFormat::JSON) {
                // Use JSON formatter for JSON topology output
                auto jsonFormatter = Formatter::create(OutputFormat::JSONPretty);
                *outStream << jsonFormatter->formatTopology(topology) << std::endl;
            } else if (config.output.topoFormat == TopoFormat::DOT) {
                *outStream << TopoVisualizer::toDot(topology) << std::endl;
            } else if (config.output.topoFormat == TopoFormat::Tree) {
                *outStream << TopoVisualizer::toTree(topology) << std::endl;
            } else {
                *outStream << TopoVisualizer::toMatrix(topology) << std::endl;
            }

            engine.finalize();
            g_engine = nullptr;
            return 0;
        }

        // Run benchmark
        if (config.output.verbose && engine.rank() == 0) {
            std::cerr << "Starting benchmark..." << std::endl;
        }

        // Set progress callback for verbose mode
        if (config.output.verbose) {
            engine.setProgressCallback([](const IntervalReport& report) {
                std::cerr << "  Progress: "
                          << formatSize(report.bytesTransferred) << " @ "
                          << report.currentBandwidthGBps << " GB/s"
                          << std::endl;
            });
        }

        auto results = engine.run();

        // Only rank 0 outputs results
        if (engine.rank() == 0) {
            *outStream << formatter->formatResults(results) << std::endl;
        }

        engine.finalize();
        g_engine = nullptr;

        // Return non-zero if verification failed
        if (config.benchmark.verifyMode != VerifyMode::None && !results.allVerified) {
            return 2;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (g_engine) {
            g_engine->finalize();
            g_engine = nullptr;
        }
        return 1;
    }
}
