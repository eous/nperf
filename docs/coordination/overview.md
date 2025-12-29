# Coordination Modes Overview

nperf supports four coordination modes for different deployment scenarios. This guide helps you choose the right mode for your environment.

## Mode Comparison

| Aspect | Local | MPI | Socket | NCCL Bootstrap |
|--------|-------|-----|--------|----------------|
| **Scope** | Single-node | Multi-node | Multi-node | Multi-node |
| **Dependencies** | CUDA only | MPI libraries | None | NCCL only |
| **Rank Discovery** | Automatic | Automatic | Automatic | Manual |
| **Job Scheduler** | None | SLURM/PBS | Custom scripts | Custom/K8s |
| **Setup Complexity** | Minimal | Moderate | Simple | Moderate |
| **Best For** | Workstations, DGX | HPC clusters | Cloud VMs | Kubernetes |

## Decision Flowchart

```
Start
  │
  ├─ Single node? ──Yes──> Local Mode
  │
  └─ Multi-node?
       │
       ├─ MPI installed? ──Yes──> MPI Mode
       │
       └─ No MPI?
            │
            ├─ Kubernetes? ──Yes──> NCCL Bootstrap Mode
            │
            └─ Cloud VMs? ──Yes──> Socket Mode
```

## Quick Selection Guide

### Use Local Mode When:
- Testing on a single machine with 1+ GPUs
- Developing and debugging
- Running on DGX or workstation
- No network coordination needed

### Use MPI Mode When:
- Running on HPC cluster with MPI
- Using SLURM or PBS job scheduler
- MPI is already configured and optimized
- Production multi-node deployment

### Use Socket Mode When:
- Multi-node without MPI installation
- Cloud VMs (AWS, GCP, Azure)
- Quick testing across nodes
- Custom orchestration

### Use NCCL Bootstrap Mode When:
- Kubernetes deployment
- Docker/container environments
- NCCL available but no MPI
- Cloud-native GPU clusters

## Feature Matrix

| Feature | Local | MPI | Socket | NCCL Bootstrap |
|---------|-------|-----|--------|----------------|
| Auto GPU detection | Yes | Yes | Yes | Yes |
| Auto rank assignment | Yes | Yes | Yes | No (manual) |
| Barrier sync | No-op | MPI_Barrier | TCP-based | TCP-based |
| NCCL ID broadcast | Local | MPI_Bcast | TCP relay | TCP relay |
| AllReduce for stats | No-op | MPI_Allreduce | TCP | TCP |
| Connection timeout | N/A | MPI-dependent | 30 seconds | 30 seconds |
| Port configuration | N/A | N/A | Configurable | Via env var |

## Mode-Specific Documentation

- [Local Mode](local.md) - Single-node multi-GPU
- [MPI Mode](mpi.md) - Multi-node with MPI
- [Socket Mode](socket.md) - Multi-node without MPI
- [NCCL Bootstrap Mode](nccl-bootstrap.md) - Kubernetes/containers

## Architecture Overview

All coordination modes implement the same `Coordinator` interface:

```
┌─────────────────────────────────────────────────────────────┐
│                    BenchmarkEngine                          │
├─────────────────────────────────────────────────────────────┤
│                    Coordinator Interface                    │
│  - initialize()      - barrier()                            │
│  - finalize()        - broadcastNcclId()                    │
│  - getRank()         - broadcast()                          │
│  - getWorldSize()    - allReduceSum()                       │
│  - getHostname()     - gather()                             │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │  Local  │   │   MPI   │   │ Socket  │   │Bootstrap│
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## NCCL ID Distribution

Each mode handles NCCL unique ID distribution differently:

1. **Local**: Generates ID directly with `ncclGetUniqueId()`
2. **MPI**: Broadcasts via `MPI_Bcast()`
3. **Socket**: Rank 0 generates, broadcasts via TCP
4. **NCCL Bootstrap**: Rank 0 generates, broadcasts via TCP

All modes ensure all ranks receive the same NCCL unique ID before communicator initialization.
