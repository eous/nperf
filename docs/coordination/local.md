# Local Mode

Local mode is for single-node, multi-GPU benchmarking. It's the simplest coordination mode with no network dependencies.

## Overview

- **Use Case**: Single machine with 1+ GPUs
- **Dependencies**: CUDA only
- **Rank**: Always 0
- **World Size**: 1 (single process controls all GPUs)

## How It Works

In local mode:
1. nperf detects all available GPUs via CUDA
2. Creates a single NCCL communicator spanning selected GPUs
3. No inter-process synchronization needed
4. All operations complete within one process

## CLI Options

```bash
./nperf --local [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--local` | (default) | Enable local mode |
| `-n, --num-gpus N` | All available | Number of GPUs to use |

## Examples

### Use All Available GPUs
```bash
./nperf --op allreduce -b 1K -B 1G
```

### Specific GPU Count
```bash
./nperf -n 4 --op allreduce -b 1M
```

### Single GPU (Baseline)
```bash
./nperf -n 1 --op allreduce -b 1M
```

### Full Benchmark Suite
```bash
./nperf --local -n 8 --op allreduce -b 1K -B 1G -i 100 --verify
```

## GPU Selection

By default, nperf uses all GPUs visible to CUDA. Control GPU visibility with:

```bash
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./nperf --op allreduce -b 1M

# Use GPUs 2-5
CUDA_VISIBLE_DEVICES=2,3,4,5 ./nperf --op allreduce -b 1M
```

Or specify count directly:
```bash
./nperf -n 4 --op allreduce -b 1M
```

## When to Use Local Mode

**Recommended for:**
- Development and debugging
- Single-node DGX systems
- Workstation testing
- Quick performance checks
- Baseline measurements

**Not suitable for:**
- Multi-node clusters
- Distributed benchmarking
- Cross-node NCCL testing

## Performance Considerations

- Local mode has minimal coordination overhead
- All GPUs share the same PCIe/NVLink fabric
- Best for measuring intra-node bandwidth
- NVLink topology significantly affects results

## Viewing Topology

Before benchmarking, view your GPU topology:
```bash
./nperf --topology
```

Example output:
```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NV4     NV4     NV4
GPU1    NV4      X      NV4     NV4
GPU2    NV4     NV4      X      NV4
GPU3    NV4     NV4     NV4      X
```

## Troubleshooting

### No GPUs Detected
```bash
# Check CUDA is working
nvidia-smi

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Fewer GPUs Than Expected
```bash
# Verify GPU count
./nperf --topology

# Check for busy GPUs
nvidia-smi
```

## See Also

- [Coordination Overview](overview.md)
- [Getting Started](../getting-started.md)
- [Topology Visualization](../output/topology.md)
