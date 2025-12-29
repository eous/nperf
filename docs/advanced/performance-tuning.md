# Performance Tuning

This guide covers advanced performance tuning options in nperf for optimizing NCCL benchmarks.

## Algorithm Selection

NCCL provides multiple algorithms for collective operations. nperf allows you to force specific algorithms.

### Available Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `auto` | NCCL auto-selection (default) | General use |
| `ring` | Ring-based | Large messages, high bandwidth |
| `tree` | Tree-based | Small messages, low latency |
| `collnetdirect` | Direct CollNet | InfiniBand Sharp |
| `collnetchain` | Chain CollNet | InfiniBand Sharp |
| `nvls` | NVLink SHARP | NVSwitch systems |

**Note**: `collnetdirect` and `collnetchain` require InfiniBand with SHARP support. `nvls` requires NVSwitch (DGX A100/H100). If the required hardware is not present, NCCL will fall back to `auto`.

### Specifying Algorithm

```bash
./nperf --op allreduce --algo ring -b 1M -B 1G
./nperf --op allreduce --algo tree -b 1K -B 1M
```

### Algorithm Comparison

```bash
# Compare algorithms
for algo in auto ring tree; do
    echo "=== $algo ==="
    ./nperf --op allreduce --algo $algo -b 256M -i 50
done
```

## Protocol Selection

NCCL uses different protocols for different message sizes.

### Available Protocols

| Protocol | Description | Best For |
|----------|-------------|----------|
| `auto` | NCCL auto-selection (default) | General use |
| `simple` | Simple protocol | Large messages |
| `ll` | Low-latency | Very small messages |
| `ll128` | Low-latency 128B | Small messages |

### Specifying Protocol

```bash
./nperf --op allreduce --proto simple -b 1G
./nperf --op allreduce --proto ll -b 1K
```

### Protocol Comparison

```bash
# Compare protocols
for proto in auto simple ll ll128; do
    echo "=== $proto ==="
    ./nperf --op allreduce --proto $proto -b 64K -i 100
done
```

## Iteration and Timing

### Iteration-Based Benchmarking

```bash
# Default: 20 iterations per message size
./nperf --op allreduce -b 1M

# More iterations for stable results
./nperf --op allreduce -b 1M -i 100

# High precision measurement
./nperf --op allreduce -b 1M -i 1000
```

### Time-Based Benchmarking

Run for a specific duration instead of fixed iterations:

```bash
# Run each message size for 5 seconds
./nperf --op allreduce -b 1K -B 1G -t 5
```

### Warmup Iterations

Warmup allows GPU to reach steady state:

```bash
# Default: 5 warmup iterations
./nperf --op allreduce -b 1M

# More warmup for cold GPUs
./nperf --op allreduce -b 1M -w 20

# Skip warmup (not recommended)
./nperf --op allreduce -b 1M -w 0
```

### Omit Initial Results

Exclude early measurements that may include initialization overhead:

```bash
# Omit first 2 seconds
./nperf --op allreduce -b 1M -t 10 -O 2
```

## Message Size Sweeps

### Size Range

```bash
# Single size
./nperf --op allreduce -b 1M

# Range with default step (2x)
./nperf --op allreduce -b 1K -B 1G

# Custom step factor
./nperf --op allreduce -b 1K -B 1G -S 4
```

### Finding Optimal Message Size

```bash
# Fine-grained sweep around expected optimum
./nperf --op allreduce -b 128M -B 512M -S 1.5 -i 100
```

## GPU Configuration

### GPU Selection

```bash
# Use specific GPU count
./nperf -n 4 --op allreduce -b 1M

# Use specific GPUs via CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,4,5 ./nperf --op allreduce -b 1M
```

### GPU Device ID

```bash
# Force specific device (local mode)
./nperf --device 0 --op allreduce -b 1M
```

## CUDA Optimization

### CUDA Graphs

Enable CUDA graphs for reduced launch overhead:

```bash
./nperf --op allreduce --graph -b 1M -i 1000
```

### Stream Synchronization

nperf uses CUDA events for precise timing. No additional configuration needed.

## NCCL Tuning via Environment

### Algorithm/Protocol Override

```bash
# Force ring algorithm globally
export NCCL_ALGO=Ring
./nperf --op allreduce -b 1M

# Force simple protocol
export NCCL_PROTO=Simple
./nperf --op allreduce -b 1M
```

### Network Tuning

```bash
# Enable InfiniBand
export NCCL_IB_DISABLE=0

# Enable GPUDirect RDMA
export NCCL_NET_GDR_LEVEL=2

# Tune buffer sizes
export NCCL_BUFFSIZE=4194304
```

### Threading

```bash
# NCCL threading model
export NCCL_NTHREADS=512

# Launch threads per ring
export NCCL_MAX_NCHANNELS=32
```

## Performance Analysis

### Baseline Measurement

```bash
# Establish baseline
./nperf --op allreduce -b 1K -B 1G -i 100 -J -o baseline.json
```

### A/B Comparison

```bash
# Test A: Default settings
./nperf --op allreduce -b 256M -i 100 -J -o test_a.json

# Test B: Ring algorithm
./nperf --op allreduce --algo ring -b 256M -i 100 -J -o test_b.json

# Compare
jq -s '.[0].results[0].bandwidth, .[1].results[0].bandwidth' test_a.json test_b.json
```

### Statistical Analysis

```bash
# Multiple runs for statistics
for i in {1..10}; do
    ./nperf --op allreduce -b 256M -i 50 -J >> results.jsonl
done

# Analyze variance
jq -s '[.[].results[0].bandwidth] | {mean: (add/length), min: min, max: max}' results.jsonl
```

## Best Practices

### For Latency Testing

- Use small message sizes (1K-64K)
- Many iterations (1000+)
- Consider CUDA graphs
- Use `ll` or `ll128` protocol

```bash
./nperf --op allreduce --proto ll --graph -b 1K -i 10000
```

### For Bandwidth Testing

- Use large message sizes (256M-1G)
- Moderate iterations (50-100)
- Use `ring` algorithm, `simple` protocol

```bash
./nperf --op allreduce --algo ring --proto simple -b 1G -i 100
```

### For Production Benchmarks

- Multiple runs for stability
- Document environment variables
- Save JSON output
- Include topology information

```bash
./nperf --topology -J -o topology.json
./nperf --op allreduce -b 1K -B 1G -i 100 -J -o results.json
```

## Tuning Checklist

1. **Verify topology**: `./nperf --topology`
2. **Establish baseline**: Default settings, multiple sizes
3. **Test algorithms**: Compare ring, tree for your sizes
4. **Test protocols**: Compare ll, ll128, simple
5. **Optimize iterations**: Find stable result count
6. **Document settings**: Save configuration and results

## Common Tuning Scenarios

### DGX A100 (8 GPUs, NVSwitch)

```bash
export NCCL_ALGO=NVLSTree
./nperf --op allreduce -b 1K -B 1G -i 100
```

### Multi-Node InfiniBand

```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
mpirun -np 16 ./nperf --mpi --op allreduce -b 1K -B 1G
```

### Cloud VMs (TCP)

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
./nperf -s -n 3 --op allreduce -b 1M -B 1G
```

## See Also

- [Environment Variables](environment-variables.md)
- [Troubleshooting](troubleshooting.md)
- [CLI Reference](../cli-reference.md)
