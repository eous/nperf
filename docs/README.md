# nperf Documentation

**nperf** is a high-performance NCCL benchmarking utility for GPU clusters. It provides comprehensive performance analysis of NCCL collective operations with support for single-node and multi-node configurations.

## Features

- **9 Collective Operations**: AllReduce, AllGather, Broadcast, Reduce, ReduceScatter, AlltoAll, Gather, Scatter, SendRecv
- **10 Data Types**: Float32, Float64, Float16, BFloat16, Int8, UInt8, Int32, UInt32, Int64, UInt64
- **4 Coordination Modes**: Local, MPI, Socket, NCCL Bootstrap
- **Rich Output**: Text tables, JSON, topology visualization
- **Correctness Verification**: Per-iteration or post-benchmark validation
- **CUDA Graph Support**: Reduced kernel launch overhead

---

## Quick Links

### Getting Started
- [Quick Start Guide](getting-started.md) - Installation and first benchmark
- [CLI Reference](cli-reference.md) - Complete command-line options

### Coordination Modes
- [Coordination Overview](coordination/overview.md) - Choosing the right mode
- [Local Mode](coordination/local.md) - Single-node multi-GPU
- [MPI Mode](coordination/mpi.md) - Multi-node with MPI
- [Socket Mode](coordination/socket.md) - Multi-node without MPI
- [NCCL Bootstrap](coordination/nccl-bootstrap.md) - Kubernetes/containers

### Benchmarking
- [Collective Operations](benchmarking/collective-operations.md) - All supported operations
- [Data Types](benchmarking/data-types.md) - Supported data types
- [Verification](benchmarking/verification.md) - Correctness checking
- [CUDA Graphs](benchmarking/cuda-graphs.md) - Performance optimization

### Output
- [Output Formats](output/formats.md) - JSON and text output
- [Topology Visualization](output/topology.md) - GPU topology display

### Advanced
- [Performance Tuning](advanced/performance-tuning.md) - Algorithm and protocol selection
- [Environment Variables](advanced/environment-variables.md) - NCCL configuration
- [Troubleshooting](advanced/troubleshooting.md) - Common issues

### Examples
- [Single-Node Examples](examples/single-node.md)
- [Multi-Node MPI Examples](examples/multi-node-mpi.md)
- [Multi-Node Socket Examples](examples/multi-node-socket.md)
- [Kubernetes Examples](examples/kubernetes.md)

---

## Quick Example

```bash
# Single-node benchmark with all GPUs
./nperf --op allreduce -b 1K -B 1G

# Multi-node with MPI (8 ranks)
mpirun -np 8 ./nperf --mpi --op allreduce -b 1M -B 1G

# View topology only
./nperf --topology
```

---

## Requirements

- CUDA Toolkit 11.0+
- NCCL 2.x
- CMake 3.18+
- C++17 compiler
- (Optional) MPI for multi-node coordination
