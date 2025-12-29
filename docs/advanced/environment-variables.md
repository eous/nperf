# Environment Variables

This guide covers environment variables that affect nperf and NCCL behavior.

## NCCL Environment Variables

### Debugging

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_DEBUG` | OFF, VERSION, WARN, INFO, TRACE | Debug output level |
| `NCCL_DEBUG_FILE` | path | Write debug to file |
| `NCCL_DEBUG_SUBSYS` | INIT, COLL, P2P, ... | Debug specific subsystems |

```bash
# Basic debug info
export NCCL_DEBUG=INFO
./nperf --op allreduce -b 1M

# Detailed tracing
export NCCL_DEBUG=TRACE
./nperf --op allreduce -b 1M

# Write to file
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/tmp/nccl_debug.%h.%p.log
./nperf --op allreduce -b 1M
```

### Algorithm and Protocol

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_ALGO` | Ring, Tree, CollNetDirect, CollNetChain, NVLS | Force algorithm |
| `NCCL_PROTO` | Simple, LL, LL128 | Force protocol |
| `NCCL_GRAPH_MIXING_SUPPORT` | 0, 1 | Graph capture support |

```bash
# Force ring algorithm
export NCCL_ALGO=Ring
./nperf --op allreduce -b 256M

# Force tree with LL128
export NCCL_ALGO=Tree
export NCCL_PROTO=LL128
./nperf --op allreduce -b 64K
```

### Network Configuration

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_IB_DISABLE` | 0, 1 | Disable InfiniBand |
| `NCCL_NET_GDR_LEVEL` | 0-5 | GPUDirect RDMA level |
| `NCCL_NET_GDR_READ` | 0, 1 | Use GDR for reads |
| `NCCL_IB_HCA` | device_name | Select IB device |
| `NCCL_IB_GID_INDEX` | 0-N | RoCE GID index |
| `NCCL_SOCKET_IFNAME` | eth0, ib0, ... | Network interface |
| `NCCL_SOCKET_NTHREADS` | 1-N | Socket threads |

```bash
# Disable InfiniBand (force TCP)
export NCCL_IB_DISABLE=1
./nperf -s -n 3 --op allreduce -b 1M

# Enable GPUDirect RDMA
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
./nperf -s -n 3 --op allreduce -b 1M

# Select specific IB device
export NCCL_IB_HCA=mlx5_0
./nperf -s -n 3 --op allreduce -b 1M
```

### P2P and NVLink

| Variable | Values | Description |
|----------|--------|-------------|
| `NCCL_P2P_DISABLE` | 0, 1 | Disable P2P |
| `NCCL_P2P_LEVEL` | 0-5 | P2P topology level |
| `NCCL_SHM_DISABLE` | 0, 1 | Disable shared memory |
| `NCCL_NVLS_ENABLE` | 0, 1 | Enable NVLink SHARP |

```bash
# Force P2P through CPU
export NCCL_P2P_DISABLE=1
./nperf --op allreduce -b 1M

# Enable NVLS on NVSwitch systems
export NCCL_NVLS_ENABLE=1
./nperf --op allreduce -b 1M
```

### Buffer and Threading

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_BUFFSIZE` | 4194304 | Buffer size (bytes) |
| `NCCL_NTHREADS` | 512 | Threads per block |
| `NCCL_MAX_NCHANNELS` | 32 | Max channels |
| `NCCL_MIN_NCHANNELS` | 1 | Min channels |
| `NCCL_NCHANNELS_PER_NET_PEER` | 1 | Channels per network peer |

```bash
# Increase buffer size
export NCCL_BUFFSIZE=8388608
./nperf --op allreduce -b 1G

# More channels
export NCCL_MAX_NCHANNELS=64
./nperf --op allreduce -b 1G
```

### Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_TIMEOUT` | 300 | Operation timeout (seconds) |
| `NCCL_COMM_BLOCKING` | 0, 1 | Blocking communicator |

```bash
# Increase timeout for slow networks
export NCCL_TIMEOUT=1800
mpirun -np 16 ./nperf --mpi --op allreduce -b 1G
```

### Bootstrap (for NCCL Bootstrap Mode)

| Variable | Format | Description |
|----------|--------|-------------|
| `NCCL_COMM_ID` | host:port | Bootstrap address |

```bash
export NCCL_COMM_ID=node0:5201
./nperf --nccl-bootstrap --rank 0 --world-size 4 --op allreduce -b 1M
```

## CUDA Environment Variables

### Device Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `CUDA_VISIBLE_DEVICES` | 0,1,2,... | Visible GPU IDs |
| `CUDA_DEVICE_ORDER` | FASTEST_FIRST, PCI_BUS_ID | Device ordering |

```bash
# Use only GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2
./nperf --op allreduce -b 1M

# Order by PCI bus (consistent ordering)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
./nperf --op allreduce -b 1M
```

### Memory

| Variable | Values | Description |
|----------|--------|-------------|
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | 0, 1 | Force device allocation |

## MPI Environment Variables

These are typically set by MPI launchers:

| Variable | Description |
|----------|-------------|
| `OMPI_COMM_WORLD_RANK` | Rank (OpenMPI) |
| `OMPI_COMM_WORLD_SIZE` | World size (OpenMPI) |
| `PMI_RANK` | Rank (SLURM) |
| `PMI_SIZE` | World size (SLURM) |
| `MPI_LOCALRANKID` | Local rank |

## Common Configurations

### Development/Debug

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
./nperf --op allreduce -b 1M
```

### High Performance (InfiniBand)

```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1
export NCCL_IB_CUDA_SUPPORT=1
mpirun -np 8 ./nperf --mpi --op allreduce -b 1G
```

### Cloud/Ethernet

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_NTHREADS=4
./nperf -s -n 3 --op allreduce -b 1M
```

### DGX A100/H100 (NVSwitch)

```bash
export NCCL_NVLS_ENABLE=1
export NCCL_ALGO=NVLSTree
./nperf -n 8 --op allreduce -b 1G
```

### Debugging Network Issues

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export NCCL_IB_SHOW_DEVICES=1
./nperf -s -n 1 --op allreduce -b 1M
```

## Environment Scripts

### save_env.sh

```bash
#!/bin/bash
# Save current NCCL environment
env | grep -E '^(NCCL_|CUDA_)' > nccl_env.txt
echo "Environment saved to nccl_env.txt"
```

### load_env.sh

```bash
#!/bin/bash
# Load NCCL environment
source nccl_env.txt
echo "Environment loaded"
```

### benchmark_with_env.sh

```bash
#!/bin/bash
# Run benchmark with specific environment

# Base settings
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Network settings
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

# Run benchmark
./nperf "$@"
```

## Precedence

When settings conflict:

1. **CLI flags** (highest priority)
2. **Environment variables**
3. **NCCL defaults** (lowest priority)

Example:
```bash
export NCCL_ALGO=Ring
./nperf --op allreduce --algo tree -b 1M
# Uses tree (CLI overrides env)
```

## Checking Current Settings

```bash
# Show NCCL settings
env | grep NCCL

# Show CUDA settings
env | grep CUDA

# Show in debug output
NCCL_DEBUG=INFO ./nperf --op allreduce -b 1M 2>&1 | head -50
```

## See Also

- [Performance Tuning](performance-tuning.md)
- [Troubleshooting](troubleshooting.md)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
