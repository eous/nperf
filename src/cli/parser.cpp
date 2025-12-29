#include "nperf/cli/parser.h"
#include "nperf/version.h"
#include <CLI/CLI.hpp>
#include <iostream>

namespace nperf {

ArgParser::ArgParser() = default;

bool ArgParser::parse(int argc, char** argv) {
    CLI::App app{NPERF_VERSION_STRING " - NCCL Performance Benchmark"};

    // Mode selection
    bool serverMode = false;
    bool mpiMode = false;
    bool localMode = false;
    bool ncclBootstrap = false;
    std::string clientHost;

    app.add_flag("-s,--server", serverMode, "Run in server mode (socket coordination)");
    app.add_option("-c,--client", clientHost, "Connect to server (socket coordination)");
    app.add_flag("--mpi", mpiMode, "Use MPI coordination");
    app.add_flag("--local", localMode, "Single-node local mode");
    app.add_flag("--nccl-bootstrap", ncclBootstrap, "Use NCCL native bootstrap (requires NCCL_COMM_ID)");

    // Collective operation
    std::string opStr = "allreduce";
    app.add_option("--op", opStr, "Collective operation")->check(
        CLI::IsMember({"allreduce", "allgather", "broadcast", "reduce",
                       "reducescatter", "alltoall", "gather", "scatter", "sendrecv"}));

    // Message size
    std::string minBytesStr = "1K";
    std::string maxBytesStr;
    size_t stepFactor = 2;
    app.add_option("-b,--bytes", minBytesStr, "Minimum message size (supports K/M/G)");
    app.add_option("-B,--max-bytes", maxBytesStr, "Maximum message size");
    app.add_option("-S,--step", stepFactor, "Size step factor");

    // Duration
    int iterations = 20;
    double testTime = 0.0;
    app.add_option("-i,--iters", iterations, "Number of iterations per size");
    app.add_option("-t,--time", testTime, "Test duration in seconds (time-based mode)");

    // Warmup
    int warmup = 5;
    double omit = 0.0;
    app.add_option("-w,--warmup", warmup, "Warmup iterations");
    app.add_option("-O,--omit", omit, "Omit seconds from start");

    // Output
    bool jsonOutput = false;
    std::string outputFile;
    double interval = 1.0;
    app.add_flag("-J,--json", jsonOutput, "JSON output format");
    app.add_option("-o,--output", outputFile, "Output file (default: stdout)");
    app.add_option("--interval", interval, "Progress report interval (seconds)");

    // Topology
    bool topologyOnly = false;
    std::string topoFormatStr = "matrix";
    bool showTransport = false;
    app.add_flag("--topology", topologyOnly, "Show topology only, no benchmark");
    app.add_option("--topo-format", topoFormatStr, "Topology format")
       ->check(CLI::IsMember({"matrix", "tree", "dot", "json"}));
    app.add_flag("--show-transport", showTransport, "Show detected transport");

    // Data type and reduction
    std::string dtypeStr = "float32";
    std::string redopStr = "sum";
    app.add_option("--dtype", dtypeStr, "Data type")
       ->check(CLI::IsMember({"float32", "float64", "float16", "bfloat16",
                              "int8", "uint8", "int32", "uint32", "int64", "uint64"}));
    app.add_option("--redop", redopStr, "Reduction operation")
       ->check(CLI::IsMember({"sum", "prod", "min", "max", "avg"}));

    // Root rank for rooted operations
    int rootRank = 0;
    app.add_option("--root", rootRank, "Root rank for broadcast/reduce/gather/scatter");

    // Algorithm and protocol
    std::string algoStr = "auto";
    std::string protoStr = "auto";
    app.add_option("--algo", algoStr, "NCCL algorithm")
       ->check(CLI::IsMember({"auto", "ring", "tree", "collnetdirect", "collnetchain", "nvls"}));
    app.add_option("--proto", protoStr, "NCCL protocol")
       ->check(CLI::IsMember({"auto", "simple", "ll", "ll128"}));

    // CUDA options
    bool cudaGraph = false;
    int cudaDevice = -1;
    app.add_flag("--graph", cudaGraph, "Enable CUDA Graph capture mode");
    app.add_option("--device", cudaDevice, "CUDA device ID (-1 = auto)");

    // Verification
    bool verify = false;
    double verifyTol = 1e-5;
    app.add_flag("--verify", verify, "Enable per-iteration verification");
    app.add_option("--verify-tolerance", verifyTol, "Verification tolerance");

    // Socket options
    int port = 5201;
    int numClients = 1;
    app.add_option("-p,--port", port, "Socket port");
    app.add_option("-n,--num-gpus", numClients, "Number of GPUs/clients");

    // NCCL bootstrap options
    int rank = -1;
    int worldSize = -1;
    app.add_option("--rank", rank, "Rank for NCCL bootstrap mode");
    app.add_option("--world-size", worldSize, "World size for NCCL bootstrap mode");

    // Misc
    bool verbose = false;
    bool debug = false;
    app.add_flag("-v,--verbose", verbose, "Verbose output");
    app.add_flag("--debug", debug, "Enable NCCL debug output");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        if (e.get_exit_code() == static_cast<int>(CLI::ExitCodes::Success)) {
            helpRequested_ = true;
            return false;
        }
        errorMessage_ = e.what();
        return false;
    }

    // Build config from parsed values
    config_.benchmark.operation = parseCollectiveOp(opStr);
    config_.benchmark.dataType = parseDataType(dtypeStr);
    config_.benchmark.reduceOp = parseReduceOp(redopStr);

    config_.benchmark.minBytes = parseSize(minBytesStr);
    if (maxBytesStr.empty()) {
        config_.benchmark.maxBytes = config_.benchmark.minBytes;
    } else {
        config_.benchmark.maxBytes = parseSize(maxBytesStr);
    }
    config_.benchmark.stepFactor = stepFactor;

    if (testTime > 0) {
        config_.benchmark.useTimeBased = true;
        config_.benchmark.testDurationSeconds = testTime;
    } else {
        config_.benchmark.iterations = iterations;
    }
    config_.benchmark.warmupIterations = warmup;
    config_.benchmark.omitSeconds = omit;

    config_.benchmark.useCudaGraph = cudaGraph;
    config_.benchmark.cudaDevice = cudaDevice;

    if (verify) {
        config_.benchmark.verifyMode = VerifyMode::PerIteration;
        config_.benchmark.verifyTolerance = verifyTol;
    }

    config_.benchmark.rootRank = rootRank;

    // Algorithm
    if (algoStr == "ring") config_.benchmark.algorithm = Algorithm::Ring;
    else if (algoStr == "tree") config_.benchmark.algorithm = Algorithm::Tree;
    else if (algoStr == "collnetdirect") config_.benchmark.algorithm = Algorithm::CollNetDirect;
    else if (algoStr == "collnetchain") config_.benchmark.algorithm = Algorithm::CollNetChain;
    else if (algoStr == "nvls") config_.benchmark.algorithm = Algorithm::NVLS;
    else config_.benchmark.algorithm = Algorithm::Auto;

    // Protocol
    if (protoStr == "simple") config_.benchmark.protocol = Protocol::Simple;
    else if (protoStr == "ll") config_.benchmark.protocol = Protocol::LL;
    else if (protoStr == "ll128") config_.benchmark.protocol = Protocol::LL128;
    else config_.benchmark.protocol = Protocol::Auto;

    // Coordination mode
    if (mpiMode) {
        config_.coordination.mode = CoordinationMode::MPI;
    } else if (ncclBootstrap) {
        config_.coordination.mode = CoordinationMode::NcclBootstrap;
        config_.coordination.rank = rank;
        config_.coordination.worldSize = worldSize;
    } else if (serverMode) {
        config_.coordination.mode = CoordinationMode::Socket;
        config_.coordination.isServer = true;
        config_.coordination.port = port;
        config_.coordination.expectedClients = numClients;
    } else if (!clientHost.empty()) {
        config_.coordination.mode = CoordinationMode::Socket;
        config_.coordination.isServer = false;
        config_.coordination.serverHost = clientHost;
        config_.coordination.port = port;
    } else {
        config_.coordination.mode = CoordinationMode::Local;
        config_.coordination.numLocalGpus = numClients;
    }

    // Output
    config_.output.format = jsonOutput ? OutputFormat::JSONPretty : OutputFormat::Text;
    config_.output.outputFile = outputFile;
    config_.output.topologyOnly = topologyOnly;
    config_.output.showTransport = showTransport;
    config_.output.verbose = verbose;
    config_.output.debug = debug;

    // Topology format
    if (topoFormatStr == "tree") config_.output.topoFormat = TopoFormat::Tree;
    else if (topoFormatStr == "dot") config_.output.topoFormat = TopoFormat::DOT;
    else if (topoFormatStr == "json") config_.output.topoFormat = TopoFormat::JSON;
    else config_.output.topoFormat = TopoFormat::Matrix;

    // Validate
    std::string validationError;
    if (!config_.validate(validationError)) {
        errorMessage_ = validationError;
        return false;
    }

    return true;
}

void printUsage() {
    std::cout << R"(
Usage: nperf [OPTIONS]

NCCL Performance Benchmark Tool

Modes:
  -s, --server            Server mode (socket coordination)
  -c, --client HOST       Client mode, connect to HOST
  --mpi                   MPI coordination mode
  --local                 Single-node local mode (default)
  --nccl-bootstrap        NCCL native bootstrap (uses NCCL_COMM_ID env var)
    --rank N              Rank ID (required with --nccl-bootstrap)
    --world-size N        Total ranks (required with --nccl-bootstrap)

Collective Operations:
  --op OPERATION          Collective: allreduce, allgather, broadcast,
                          reduce, reducescatter, alltoall, gather,
                          scatter, sendrecv (default: allreduce)
  --root N                Root rank for rooted ops (default: 0)

Message Size:
  -b, --bytes SIZE        Minimum message size (default: 1K)
  -B, --max-bytes SIZE    Maximum message size (default: same as -b)
  -S, --step FACTOR       Size step factor (default: 2)

Duration:
  -i, --iters N           Iterations per size (default: 20)
  -t, --time SECONDS      Time-based duration mode
  -w, --warmup N          Warmup iterations (default: 5)
  -O, --omit SECONDS      Omit seconds from start

Output:
  -J, --json              JSON output format
  -o, --output FILE       Output to file
  --interval SECONDS      Progress report interval

Topology:
  --topology              Show topology only
  --topo-format FORMAT    Topology format: matrix, tree, dot, json
  --show-transport        Show detected transport

Data:
  --dtype TYPE            Data type: float32, float64, float16, bfloat16,
                          int8, uint8, int32, uint32, int64, uint64
                          (default: float32)
  --redop OP              Reduction: sum, prod, min, max, avg (default: sum)

NCCL Options:
  --algo ALGO             Algorithm: auto, ring, tree, collnetdirect,
                          collnetchain, nvls (default: auto)
  --proto PROTO           Protocol: auto, simple, ll, ll128 (default: auto)
  --graph                 Enable CUDA Graph capture mode

Verification:
  --verify                Enable per-iteration verification
  --verify-tolerance TOL  Floating point tolerance (default: 1e-5)

Other:
  -p, --port PORT         Socket port (default: 5201)
  -n, --num-gpus N        Number of GPUs (local mode)
  --device ID             CUDA device ID
  -v, --verbose           Verbose output
  --debug                 Enable NCCL debug output
  -h, --help              Show this help
  --version               Show version

Examples:
  # Single-node benchmark with 8 GPUs
  nperf --local -n 8 --op allreduce -b 1K -B 1G

  # MPI multi-node benchmark
  mpirun -np 8 nperf --mpi --op allreduce -b 1M

  # Socket mode
  nperf -s -n 7 &                    # Server with 7 clients
  nperf -c server-host               # Client

  # NCCL bootstrap mode (no MPI required)
  export NCCL_COMM_ID=node0:5201
  nperf --nccl-bootstrap --rank 0 --world-size 2 &  # On node0
  nperf --nccl-bootstrap --rank 1 --world-size 2    # On node1

  # Show topology only
  nperf --topology --topo-format dot > topo.dot
)" << std::endl;
}

void printVersion() {
    std::cout << NPERF_VERSION_STRING << std::endl;
    std::cout << "Built with CUDA and NCCL support" << std::endl;
#ifdef NPERF_HAS_MPI
    std::cout << "MPI support: enabled" << std::endl;
#else
    std::cout << "MPI support: disabled" << std::endl;
#endif
}

} // namespace nperf
