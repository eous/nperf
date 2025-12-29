# Getting Started with nperf

This guide will help you install nperf and run your first NCCL benchmark.

## Prerequisites

### Required
- **CUDA Toolkit** 11.0 or later
- **NCCL** 2.x (usually included with CUDA or available separately)
- **CMake** 3.18 or later
- **C++17 compatible compiler** (GCC 7+, Clang 5+)

### Optional
- **MPI** (OpenMPI, MPICH, or Intel MPI) for multi-node benchmarking
- **NVML** (included with CUDA drivers) for topology detection

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/nperf.git
cd nperf

# Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# The binary is at build/nperf
./build/nperf --help
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `NPERF_BUILD_MPI` | ON | Enable MPI coordination support |
| `NPERF_BUILD_TESTS` | ON | Build unit and integration tests |
| `CMAKE_BUILD_TYPE` | Release | Build type (Release recommended) |

Example with options:
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNPERF_BUILD_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

## Your First Benchmark

### Single-GPU Test

```bash
./nperf --local -n 1 --op allreduce -b 1K -B 1M
```

This runs an AllReduce benchmark with message sizes from 1KB to 1MB.

### Multi-GPU Test (Same Node)

```bash
# Use all available GPUs
./nperf --op allreduce -b 1K -B 1G

# Use specific number of GPUs
./nperf -n 4 --op allreduce -b 1M
```

### Understanding the Output

```
#       Bytes  Elements   Time(us)  BusBw(GB/s)  AlgoBw(GB/s)
         1024       256       5.23        0.39         0.20
         2048       512       5.45        0.75         0.38
         4096      1024       5.67        1.44         0.72
         8192      2048       6.12        2.68         1.34
...

Summary:
  Peak Bus Bandwidth: 245.32 GB/s
  Avg Bus Bandwidth:  198.45 GB/s
  Total Data:         2.15 GB
  Total Time:         1.23 s
```

**Columns explained:**
- **Bytes**: Message size in bytes
- **Elements**: Number of data elements (bytes / data type size)
- **Time(us)**: Average latency in microseconds
- **BusBw(GB/s)**: Bus bandwidth (normalized for the collective operation)
- **AlgoBw(GB/s)**: Algorithm bandwidth (raw data / time)

## Common Use Cases

### Quick Performance Check
```bash
./nperf --op allreduce -b 1M
```

### Full Sweep (1KB to 1GB)
```bash
./nperf --op allreduce -b 1K -B 1G -i 100
```

### JSON Output for Analysis
```bash
./nperf --op allreduce -b 1M -B 1G -J -o results.json
```

### View GPU Topology
```bash
./nperf --topology
```

### Verify Correctness
```bash
./nperf --op allreduce -b 1M --verify
```

## Next Steps

- [CLI Reference](cli-reference.md) - All command-line options
- [Coordination Modes](coordination/overview.md) - Multi-node setup
- [Examples](examples/single-node.md) - More usage examples
