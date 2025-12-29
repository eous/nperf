# NCCL Bootstrap Mode

NCCL Bootstrap mode uses NCCL's native bootstrap mechanism for multi-node coordination. It's ideal for Kubernetes and container environments.

## Overview

- **Use Case**: Kubernetes, Docker, container orchestration
- **Dependencies**: NCCL only (no MPI or custom sockets)
- **Rank**: Manually specified via CLI
- **World Size**: Manually specified via CLI

## Prerequisites

1. NCCL library with bootstrap support
2. `NCCL_COMM_ID` environment variable set (or equivalent)
3. Network connectivity between all nodes on bootstrap port

## CLI Options

```bash
./nperf --nccl-bootstrap --rank N --world-size M [OPTIONS]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--nccl-bootstrap` | Yes | Enable NCCL bootstrap mode |
| `--rank N` | Yes | This node's rank (0 to world-size-1) |
| `--world-size N` | Yes | Total number of ranks |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NCCL_COMM_ID` | Bootstrap address in format `hostname:port` |

## How It Works

1. Set `NCCL_COMM_ID` to point to rank 0's address
2. Rank 0 listens for incoming connections
3. Other ranks connect to rank 0
4. NCCL handles ID distribution internally
5. All ranks create NCCL communicator
6. Benchmark runs with TCP-based barriers

## Examples

### Two Nodes

```bash
# Node 0 (rank 0)
export NCCL_COMM_ID=node0:5201
./nperf --nccl-bootstrap --rank 0 --world-size 2 --op allreduce -b 1M

# Node 1 (rank 1)
export NCCL_COMM_ID=node0:5201
./nperf --nccl-bootstrap --rank 1 --world-size 2 --op allreduce -b 1M
```

### Four Nodes

```bash
# All nodes use same NCCL_COMM_ID pointing to rank 0
export NCCL_COMM_ID=node0:5201

# Node 0
./nperf --nccl-bootstrap --rank 0 --world-size 4 --op allreduce -b 1M -B 1G

# Node 1
./nperf --nccl-bootstrap --rank 1 --world-size 4 --op allreduce -b 1M -B 1G

# Node 2
./nperf --nccl-bootstrap --rank 2 --world-size 4 --op allreduce -b 1M -B 1G

# Node 3
./nperf --nccl-bootstrap --rank 3 --world-size 4 --op allreduce -b 1M -B 1G
```

### Custom Port

```bash
# Use port 6000 instead of default
export NCCL_COMM_ID=node0:6000
./nperf --nccl-bootstrap --rank 0 --world-size 2 --op allreduce -b 1M
```

## Kubernetes Deployment

### Pod Specification

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nperf-rank-0
spec:
  containers:
  - name: nperf
    image: your-registry/nperf:latest
    env:
    - name: NCCL_COMM_ID
      value: "nperf-rank-0:5201"
    - name: RANK
      value: "0"
    - name: WORLD_SIZE
      value: "4"
    command: ["./nperf"]
    args:
    - "--nccl-bootstrap"
    - "--rank"
    - "$(RANK)"
    - "--world-size"
    - "$(WORLD_SIZE)"
    - "--op"
    - "allreduce"
    - "-b"
    - "1M"
    resources:
      limits:
        nvidia.com/gpu: 1
```

### StatefulSet for Multiple Replicas

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf
spec:
  serviceName: "nperf"
  replicas: 4
  selector:
    matchLabels:
      app: nperf
  template:
    metadata:
      labels:
        app: nperf
    spec:
      containers:
      - name: nperf
        image: your-registry/nperf:latest
        env:
        - name: NCCL_COMM_ID
          value: "nperf-0.nperf:5201"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        command: ["/bin/sh", "-c"]
        args:
        - |
          RANK=$(echo $POD_NAME | grep -o '[0-9]*$')
          ./nperf --nccl-bootstrap --rank $RANK --world-size 4 --op allreduce -b 1M
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: nperf
spec:
  clusterIP: None
  selector:
    app: nperf
  ports:
  - port: 5201
    name: nccl
```

### Headless Service for DNS Discovery

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nperf-headless
spec:
  clusterIP: None
  selector:
    app: nperf
  ports:
  - port: 5201
```

## Docker Compose Example

```yaml
version: '3.8'
services:
  rank0:
    image: nperf:latest
    environment:
      - NCCL_COMM_ID=rank0:5201
    command: ./nperf --nccl-bootstrap --rank 0 --world-size 2 --op allreduce -b 1M
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  rank1:
    image: nperf:latest
    environment:
      - NCCL_COMM_ID=rank0:5201
    command: ./nperf --nccl-bootstrap --rank 1 --world-size 2 --op allreduce -b 1M
    depends_on:
      - rank0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Startup Script

```bash
#!/bin/bash
# run_nccl_bootstrap.sh

MASTER_ADDR="${MASTER_ADDR:-node0}"
MASTER_PORT="${MASTER_PORT:-5201}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

export NCCL_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"

echo "Starting nperf with NCCL Bootstrap"
echo "  NCCL_COMM_ID: $NCCL_COMM_ID"
echo "  Rank: $RANK"
echo "  World Size: $WORLD_SIZE"

./nperf --nccl-bootstrap \
  --rank "$RANK" \
  --world-size "$WORLD_SIZE" \
  --op allreduce \
  -b 1K -B 1G \
  "$@"
```

## Troubleshooting

### Connection Timeout

```bash
# Verify NCCL_COMM_ID is set correctly on all nodes
echo $NCCL_COMM_ID

# Check network connectivity to master
nc -zv node0 5201

# Ensure rank 0 starts first
# Add sleep in scripts for non-zero ranks
```

### Rank Mismatch

```bash
# Verify each node has unique rank
# Ranks must be 0 to (world_size - 1)

# Common error: duplicate ranks
# Solution: Check environment variables
```

### NCCL Initialization Failure

```bash
# Enable NCCL debug output
export NCCL_DEBUG=INFO
./nperf --nccl-bootstrap --rank 0 --world-size 2 --op allreduce -b 1M

# Check for network interface issues
export NCCL_SOCKET_IFNAME=eth0
```

### Kubernetes-Specific Issues

```bash
# Verify DNS resolution
nslookup nperf-0.nperf

# Check pod-to-pod connectivity
kubectl exec -it nperf-1 -- nc -zv nperf-0.nperf 5201

# Ensure GPUs are available
kubectl describe pod nperf-0 | grep -A5 nvidia
```

## When to Use NCCL Bootstrap Mode

**Recommended for:**
- Kubernetes GPU clusters
- Docker Swarm deployments
- Container orchestration platforms
- Cloud-native GPU workloads
- When MPI is not available

**Consider alternatives when:**
- HPC cluster with MPI → Use [MPI Mode](mpi.md)
- Cloud VMs without orchestration → Use [Socket Mode](socket.md)
- Single node → Use [Local Mode](local.md)

## See Also

- [Coordination Overview](overview.md)
- [Kubernetes Examples](../examples/kubernetes.md)
- [Troubleshooting](../advanced/troubleshooting.md)
