# nperf

A high-performance NCCL benchmarking utility for GPU cluster validation and performance analysis. Similar to iperf for network testing, nperf provides comprehensive benchmarking of NCCL collective operations with rich topology visualization and automation-friendly output.

## Features

- **All NCCL Collectives**: AllReduce, AllGather, Broadcast, Reduce, ReduceScatter, AlltoAll, SendRecv
- **Multiple Coordination Modes**: Local (single-node), MPI (multi-node), Socket (client-server)
- **GPU Topology Detection**: NVLink, NVSwitch, PCIe hierarchy, NUMA affinity
- **Rich Output Formats**: Human-readable tables, JSON, DOT (Graphviz)
- **CUDA Graph Support**: Reduced kernel launch overhead for accurate small-message timing
- **Data Verification**: Optional per-iteration correctness checking
- **Flexible Sizing**: Iteration-based or time-based benchmarking

## Requirements

- CUDA Toolkit 11.0+
- NCCL 2.10+
- CMake 3.18+
- C++17 compiler
- Optional: MPI (OpenMPI, MPICH, or similar)

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `NPERF_BUILD_MPI` | ON | Build with MPI support |
| `NPERF_BUILD_TESTS` | ON | Build unit tests |
| `NPERF_BUILD_EXAMPLES` | ON | Build examples |

## Usage

### Single-Node Benchmarking

```bash
# Benchmark AllReduce with 8 GPUs, sizes from 1KB to 1GB
./nperf --local -n 8 --op allreduce -b 1K -B 1G

# Quick topology check
./nperf --topology

# JSON output for automation
./nperf --local -n 4 --op allgather -b 1M -J
```

### Multi-Node with MPI

```bash
# 8 ranks across nodes
mpirun -np 8 --hostfile hosts.txt ./nperf --mpi --op allreduce -b 1M -B 1G
```

### Multi-Node with Socket Coordination

```bash
# On server (expects 3 clients)
./nperf -s -n 3 --op allreduce -b 1M

# On each client
./nperf -c server-hostname --op allreduce -b 1M
```

### Topology Visualization

```bash
# Matrix format (like nvidia-smi topo -m)
./nperf --topology --topo-format matrix

# Tree format
./nperf --topology --topo-format tree

# DOT format for Graphviz
./nperf --topology --topo-format dot > topology.dot
dot -Tpng topology.dot -o topology.png

# JSON format
./nperf --topology --topo-format json
```

## Command-Line Options

### Modes
| Option | Description |
|--------|-------------|
| `-s, --server` | Server mode (socket coordination) |
| `-c, --client HOST` | Client mode, connect to HOST |
| `--mpi` | MPI coordination mode |
| `--local` | Single-node local mode (default) |

### Collective Operations
| Option | Description |
|--------|-------------|
| `--op OPERATION` | allreduce, allgather, broadcast, reduce, reducescatter, alltoall, sendrecv |

### Message Size
| Option | Description |
|--------|-------------|
| `-b, --bytes SIZE` | Minimum message size (default: 1K) |
| `-B, --max-bytes SIZE` | Maximum message size |
| `-S, --step FACTOR` | Size step factor (default: 2) |

### Duration
| Option | Description |
|--------|-------------|
| `-i, --iters N` | Iterations per size (default: 20) |
| `-t, --time SECONDS` | Time-based duration mode |
| `-w, --warmup N` | Warmup iterations (default: 5) |
| `-O, --omit SECONDS` | Omit seconds from start |

### Output
| Option | Description |
|--------|-------------|
| `-J, --json` | JSON output format |
| `-o, --output FILE` | Output to file |
| `-v, --verbose` | Verbose output |
| `--debug` | Enable NCCL debug output |

### Data Types
| Option | Description |
|--------|-------------|
| `--dtype TYPE` | float32, float64, float16, bfloat16, int32, int64 |
| `--redop OP` | sum, prod, min, max, avg |

### NCCL Options
| Option | Description |
|--------|-------------|
| `--algo ALGO` | Algorithm: auto, ring, tree |
| `--proto PROTO` | Protocol: auto, simple, ll, ll128 |
| `--graph` | Enable CUDA Graph capture mode |

### Verification
| Option | Description |
|--------|-------------|
| `--verify` | Enable per-iteration verification |
| `--verify-tolerance TOL` | Floating point tolerance (default: 1e-5) |

## Output Example

```
================================================================================
                       nperf v1.0.0 - NCCL Benchmark
================================================================================
Configuration: AllReduce | float32 | sum | 8 ranks
Topology: 8x NVIDIA A100-SXM4-80GB | NVSwitch | 4x mlx5 IB HDR
--------------------------------------------------------------------------------
     Size        Count      Time(us)   Algo BW(GB/s)   Bus BW(GB/s)    Status
--------------------------------------------------------------------------------
     1 KB          256         12.34          0.08           0.07        OK
     1 MB       262144        102.34          9.78           8.55        OK
     1 GB    268435456      89234.56         11.21           9.81        OK
--------------------------------------------------------------------------------
Summary: Peak 9.81 GB/s | Avg 8.45 GB/s
================================================================================
```

## Library Usage

nperf can also be used as a library:

```cpp
#include <nperf/nperf.h>

int main() {
    nperf::BenchmarkEngine engine;

    nperf::NperfConfig config;
    config.benchmark.operation = nperf::CollectiveOp::AllReduce;
    config.benchmark.minBytes = 1024;
    config.benchmark.maxBytes = 1024 * 1024 * 1024;

    engine.configure(config);
    engine.initialize(0, nullptr);

    auto results = engine.run();

    // Process results...
    engine.finalize();
}
```

### CMake Integration

```cmake
find_package(nperf REQUIRED)
target_link_libraries(myapp PRIVATE nperf::nperf)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- Uses [nlohmann/json](https://github.com/nlohmann/json) for JSON output
- Uses [CLI11](https://github.com/CLIUtils/CLI11) for argument parsing
