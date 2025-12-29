# Single-Node Examples

Examples for running nperf on a single machine with one or more GPUs.

## Prerequisites

- CUDA-capable GPU(s)
- CUDA toolkit installed
- NCCL library installed
- nperf built

## Quick Start

### Basic AllReduce

```bash
./nperf --op allreduce -b 1M
```

Output:
```
nperf v1.0.0 - NCCL Performance Benchmark

Configuration:
  Operation:     AllReduce
  Data Type:     float32
  Reduction:     sum
  GPUs:          4
  Iterations:    20
  Warmup:        5

Results:
  Size          Time (us)     Bandwidth (GB/s)   Bus BW (GB/s)
  1M            123.45        8.10               6.08
```

### Multiple Message Sizes

```bash
./nperf --op allreduce -b 1K -B 1G
```

Sweeps from 1KB to 1GB in 2x increments.

## GPU Configuration

### Use All Available GPUs

```bash
./nperf --op allreduce -b 1M
```

### Specific GPU Count

```bash
# Use 4 GPUs
./nperf -n 4 --op allreduce -b 1M

# Use 2 GPUs
./nperf -n 2 --op allreduce -b 1M
```

### Select Specific GPUs

```bash
# GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./nperf --op allreduce -b 1M

# GPUs 2, 3, 4, 5
CUDA_VISIBLE_DEVICES=2,3,4,5 ./nperf --op allreduce -b 1M
```

## Collective Operations

### AllReduce

```bash
./nperf --op allreduce -b 1M -B 1G
```

### AllGather

```bash
./nperf --op allgather -b 1M -B 1G
```

### Broadcast

```bash
./nperf --op broadcast --root 0 -b 1M -B 1G
```

### Reduce

```bash
./nperf --op reduce --root 0 -b 1M -B 1G
```

### ReduceScatter

```bash
./nperf --op reducescatter -b 1M -B 1G
```

### All-to-All

```bash
./nperf --op alltoall -b 1M -B 1G
```

### Compare All Operations

```bash
#!/bin/bash
for op in allreduce allgather broadcast reduce reducescatter alltoall; do
    echo "=== $op ==="
    ./nperf --op $op -b 256M -i 50
    echo
done
```

## Data Types

### Default (float32)

```bash
./nperf --op allreduce -b 1M
```

### Mixed Precision

```bash
# float16
./nperf --op allreduce --dtype float16 -b 1M -B 1G

# bfloat16
./nperf --op allreduce --dtype bfloat16 -b 1M -B 1G
```

### High Precision

```bash
./nperf --op allreduce --dtype float64 -b 1M -B 1G
```

### Compare Data Types

```bash
#!/bin/bash
for dtype in float32 float16 bfloat16 float64; do
    echo "=== $dtype ==="
    ./nperf --op allreduce --dtype $dtype -b 256M -i 50
    echo
done
```

## Reduction Operations

```bash
# Sum (default)
./nperf --op allreduce --redop sum -b 1M

# Max
./nperf --op allreduce --redop max -b 1M

# Min
./nperf --op allreduce --redop min -b 1M

# Product
./nperf --op allreduce --redop prod -b 1M
```

## Benchmarking Options

### More Iterations

```bash
# 100 iterations for stable results
./nperf --op allreduce -b 256M -i 100
```

### Extended Warmup

```bash
# 20 warmup iterations
./nperf --op allreduce -b 1M -w 20 -i 100
```

### Time-Based

```bash
# Run for 10 seconds per message size
./nperf --op allreduce -b 1K -B 1G -t 10
```

### Fine-Grained Sweep

```bash
# 1.5x step instead of 2x
./nperf --op allreduce -b 1M -B 256M -S 1.5
```

## CUDA Graphs

```bash
# Enable CUDA graphs for reduced overhead
./nperf --op allreduce --graph -b 1M -i 1000
```

## Verification

### Verify Correctness

```bash
./nperf --op allreduce --verify -b 1M -i 10
```

### Custom Tolerance

```bash
./nperf --op allreduce --verify --verify-tolerance 1e-4 -b 1M
```

## Output Options

### JSON Output

```bash
./nperf --op allreduce -b 1K -B 1G -J
```

### Save to File

```bash
./nperf --op allreduce -b 1K -B 1G -J -o results.json
```

### View Topology

```bash
./nperf --topology
```

### Topology as DOT

```bash
./nperf --topology --topo-format dot > topology.dot
dot -Tpng topology.dot -o topology.png
```

## Algorithm Selection

### Ring Algorithm

```bash
./nperf --op allreduce --algo ring -b 1G
```

### Tree Algorithm

```bash
./nperf --op allreduce --algo tree -b 64K
```

### Compare Algorithms

```bash
#!/bin/bash
for algo in auto ring tree; do
    echo "=== $algo ==="
    ./nperf --op allreduce --algo $algo -b 256M -i 50
done
```

## Complete Benchmark Scripts

### Quick Health Check

```bash
#!/bin/bash
# health_check.sh - Quick GPU communication test

echo "GPU Topology:"
./nperf --topology

echo ""
echo "AllReduce Test (256MB):"
./nperf --op allreduce -b 256M -i 20

echo ""
echo "Verification Test:"
./nperf --op allreduce --verify -b 1M -i 5
```

### Full Sweep

```bash
#!/bin/bash
# full_sweep.sh - Comprehensive benchmark

OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Topology
./nperf --topology --topo-format json > "$OUTPUT_DIR/topology.json"

# All operations
for op in allreduce allgather broadcast reduce reducescatter alltoall; do
    echo "Benchmarking $op..."
    ./nperf --op $op -b 1K -B 1G -i 50 -J -o "$OUTPUT_DIR/${op}.json"
done

echo "Results saved to $OUTPUT_DIR/"
```

### Data Type Comparison

```bash
#!/bin/bash
# dtype_comparison.sh

echo "Data Type Comparison (256MB AllReduce)"
echo "======================================="

for dtype in float32 float16 bfloat16 float64; do
    result=$(./nperf --op allreduce --dtype $dtype -b 256M -i 50 -J | jq '.results[0].bandwidth')
    printf "%-10s: %s GB/s\n" "$dtype" "$result"
done
```

### Algorithm Tuning

```bash
#!/bin/bash
# algo_tuning.sh

sizes="1K 64K 1M 16M 256M 1G"
algos="auto ring tree"

echo "Algorithm Comparison"
echo "===================="
printf "%-10s" "Size"
for algo in $algos; do
    printf "%-15s" "$algo"
done
echo ""

for size in $sizes; do
    printf "%-10s" "$size"
    for algo in $algos; do
        bw=$(./nperf --op allreduce --algo $algo -b $size -i 50 -J 2>/dev/null | jq -r '.results[0].bandwidth // "N/A"')
        printf "%-15s" "$bw"
    done
    echo ""
done
```

## Performance Analysis

### Find Peak Bandwidth

```bash
./nperf --op allreduce -b 1G -i 100 -J | jq '.results[0].bandwidth'
```

### Latency at Small Sizes

```bash
./nperf --op allreduce -b 1K -i 1000 -J | jq '.results[0].timeUs.avg'
```

### Extract All Results

```bash
./nperf --op allreduce -b 1K -B 1G -J | \
    jq -r '.results[] | "\(.bytes),\(.bandwidth),\(.busBandwidth)"'
```

## See Also

- [CLI Reference](../cli-reference.md)
- [Collective Operations](../benchmarking/collective-operations.md)
- [Performance Tuning](../advanced/performance-tuning.md)
