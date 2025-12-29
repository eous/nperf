# CUDA Graphs

CUDA Graphs allow nperf to capture and replay NCCL operations with reduced CPU overhead. This is useful for measuring peak GPU-side performance.

## Overview

CUDA Graphs capture a sequence of GPU operations into a single executable unit. Once captured, the graph can be launched with minimal CPU involvement, eliminating per-operation launch overhead.

## Benefits

| Aspect | Standard Mode | CUDA Graph Mode |
|--------|---------------|-----------------|
| CPU Launch Overhead | Per iteration | Once (capture) |
| GPU Scheduling | Dynamic | Pre-computed |
| Latency Variance | Higher | Lower |
| Best For | Realistic workloads | Peak throughput |

## Enabling CUDA Graphs

```bash
./nperf --op allreduce --graph -b 1M -B 1G
```

## CLI Option

| Option | Default | Description |
|--------|---------|-------------|
| `--graph` | Disabled | Enable CUDA Graph capture |

## How It Works

### Standard Mode (Without Graphs)

```
CPU: Launch → Launch → Launch → Launch → ...
      ↓        ↓        ↓        ↓
GPU:  Op1  →  Op2  →  Op3  →  Op4  → ...
```

Each iteration involves CPU-side work to launch the kernel.

### CUDA Graph Mode

```
Capture Phase:
  CPU: [Capture Op1, Op2, Op3] → Graph

Replay Phase:
  CPU: Launch Graph (×N iterations)
       ↓
  GPU:  [Op1, Op2, Op3] × N
```

The graph is captured once, then replayed efficiently.

## Capture Process

1. **Warmup**: Standard warmup iterations (not captured)
2. **Capture**: Record operations into graph
3. **Instantiate**: Compile graph into executable
4. **Replay**: Execute graph for timed iterations

## When to Use CUDA Graphs

### Recommended

- Measuring peak NCCL bandwidth
- Low-latency benchmarking
- Comparing against hardware limits
- Small message latency testing

### Not Recommended

- Realistic application simulation
- Verification mode (incompatible)
- First-run benchmarks
- Dynamic message sizes per iteration

## Examples

### Basic Graph Benchmark

```bash
./nperf --op allreduce --graph -b 1M -i 1000
```

### Sweep with Graphs

```bash
./nperf --op allreduce --graph -b 1K -B 1G -i 100
```

### Compare Graph vs Standard

```bash
# Standard mode
./nperf --op allreduce -b 1M -i 1000 -o standard.json -J

# Graph mode
./nperf --op allreduce --graph -b 1M -i 1000 -o graph.json -J
```

### High Iteration Count

```bash
# Many iterations to amortize capture cost
./nperf --op allreduce --graph -b 64K -i 10000
```

## Performance Comparison

Expected results (varies by hardware):

| Message Size | Standard Latency | Graph Latency | Speedup |
|--------------|------------------|---------------|---------|
| 1 KB | ~15 µs | ~8 µs | 1.9x |
| 64 KB | ~20 µs | ~12 µs | 1.7x |
| 1 MB | ~50 µs | ~45 µs | 1.1x |
| 256 MB | ~5 ms | ~5 ms | ~1x |

**Key insight**: Graphs provide the biggest benefit for small messages where launch overhead dominates.

## Limitations

### No Dynamic Parameters

Graph parameters are fixed at capture time:
- Buffer pointers
- Message sizes
- Stream assignments

### Incompatible with Verification

```bash
# This will error or skip verification
./nperf --op allreduce --graph --verify -b 1M
```

### Capture Overhead

Graph capture adds one-time overhead:
- ~1-10 ms depending on complexity
- Amortized over many iterations

### Memory Overhead

Graphs consume additional GPU memory for:
- Graph structure
- Intermediate state

## Best Practices

1. **Use many iterations**: Amortize capture cost
   ```bash
   ./nperf --op allreduce --graph -b 1M -i 1000
   ```

2. **Compare both modes**: Understand realistic vs peak
   ```bash
   ./nperf --op allreduce -b 1M -i 100
   ./nperf --op allreduce --graph -b 1M -i 100
   ```

3. **Focus on small messages**: Largest relative benefit
   ```bash
   ./nperf --op allreduce --graph -b 1K -B 64K -i 1000
   ```

4. **Report both numbers**: When publishing results, show both modes

## Interpreting Results

### With Graphs

Results represent:
- Minimum achievable latency
- Peak bandwidth capability
- Hardware limits

### Without Graphs

Results represent:
- Realistic application performance
- Including CPU overhead
- Typical ML framework behavior

## Troubleshooting

### Graph Capture Fails

```bash
# Enable debug output
export NCCL_DEBUG=INFO
./nperf --op allreduce --graph --debug -b 1M
```

Possible causes:
- CUDA version incompatibility
- Driver issues
- Memory allocation failures

### No Performance Improvement

If graph mode shows no benefit:
- Message size may be too large (GPU-bound)
- CPU is not the bottleneck
- Already optimized launch path

### Memory Errors

If graph capture causes OOM:
- Reduce message size
- Reduce GPU count
- Check baseline memory usage

## Technical Details

### CUDA Graph API Usage

nperf uses:
- `cudaStreamBeginCapture()` / `cudaStreamEndCapture()`
- `cudaGraphInstantiate()`
- `cudaGraphLaunch()`

### Compatibility

- CUDA 10.0+: Basic support
- CUDA 11.0+: Improved NCCL integration
- CUDA 12.0+: Best performance

Check your CUDA version:
```bash
nvcc --version
nvidia-smi
```

## Example: Full Graph Analysis

```bash
#!/bin/bash
# compare_graph_modes.sh

SIZES="1K 4K 16K 64K 256K 1M 4M 16M 64M"

echo "Size,Standard_BW,Graph_BW,Speedup"

for size in $SIZES; do
    std=$(./nperf --op allreduce -b $size -i 100 -J | jq '.results[0].busBandwidth')
    graph=$(./nperf --op allreduce --graph -b $size -i 100 -J | jq '.results[0].busBandwidth')
    speedup=$(echo "scale=2; $graph / $std" | bc)
    echo "$size,$std,$graph,$speedup"
done
```

## See Also

- [Performance Tuning](../advanced/performance-tuning.md)
- [Collective Operations](collective-operations.md)
- [CLI Reference](../cli-reference.md)
