# Kubernetes Examples

Examples for running nperf in Kubernetes GPU clusters.

## Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA device plugin installed
- Network connectivity between pods
- nperf container image

## Container Image

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install NCCL
RUN apt-get update && apt-get install -y libnccl2 libnccl-dev

# Copy and build nperf
COPY . /nperf
WORKDIR /nperf
RUN mkdir build && cd build && cmake .. && make -j

# Set entrypoint
ENTRYPOINT ["/nperf/build/nperf"]
```

### Build and Push

```bash
docker build -t your-registry/nperf:latest .
docker push your-registry/nperf:latest
```

## NCCL Bootstrap Mode

NCCL Bootstrap mode is ideal for Kubernetes as it requires no MPI.

### Single Pod Test

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nperf-test
spec:
  restartPolicy: Never
  containers:
  - name: nperf
    image: your-registry/nperf:latest
    args: ["--op", "allreduce", "-b", "1M", "-n", "2"]
    resources:
      limits:
        nvidia.com/gpu: 2
```

### Two-Pod Example

```yaml
# Pod 0 (rank 0)
apiVersion: v1
kind: Pod
metadata:
  name: nperf-rank-0
  labels:
    app: nperf
spec:
  restartPolicy: Never
  containers:
  - name: nperf
    image: your-registry/nperf:latest
    env:
    - name: NCCL_COMM_ID
      value: "nperf-rank-0:5201"
    args:
    - "--nccl-bootstrap"
    - "--rank"
    - "0"
    - "--world-size"
    - "2"
    - "--op"
    - "allreduce"
    - "-b"
    - "1M"
    resources:
      limits:
        nvidia.com/gpu: 1
---
# Pod 1 (rank 1)
apiVersion: v1
kind: Pod
metadata:
  name: nperf-rank-1
  labels:
    app: nperf
spec:
  restartPolicy: Never
  containers:
  - name: nperf
    image: your-registry/nperf:latest
    env:
    - name: NCCL_COMM_ID
      value: "nperf-rank-0:5201"
    args:
    - "--nccl-bootstrap"
    - "--rank"
    - "1"
    - "--world-size"
    - "2"
    - "--op"
    - "allreduce"
    - "-b"
    - "1M"
    resources:
      limits:
        nvidia.com/gpu: 1
```

## StatefulSet Deployment

For automatic rank assignment based on pod ordinal.

### Headless Service

```yaml
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

### StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf
spec:
  serviceName: nperf
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
          # Extract rank from pod name (nperf-0 -> 0)
          RANK=$(echo $POD_NAME | grep -o '[0-9]*$')
          WORLD_SIZE=4

          # Wait for rank 0 to be ready
          if [ "$RANK" != "0" ]; then
            sleep 10
          fi

          # Run benchmark
          /nperf/build/nperf \
            --nccl-bootstrap \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            --op allreduce \
            -b 1K -B 1G
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Job-Based Deployment

For one-time benchmark runs.

### Indexed Job (Kubernetes 1.21+)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nperf-benchmark
spec:
  completions: 4
  parallelism: 4
  completionMode: Indexed
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: nperf
        image: your-registry/nperf:latest
        env:
        - name: NCCL_COMM_ID
          value: "nperf-benchmark-0:5201"
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
        command: ["/bin/sh", "-c"]
        args:
        - |
          RANK=$JOB_COMPLETION_INDEX
          WORLD_SIZE=4

          if [ "$RANK" != "0" ]; then
            sleep 15
          fi

          /nperf/build/nperf \
            --nccl-bootstrap \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            --op allreduce \
            -b 1M -B 1G
        resources:
          limits:
            nvidia.com/gpu: 1
```

## ConfigMap for Options

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nperf-config
data:
  OPERATION: "allreduce"
  MIN_BYTES: "1K"
  MAX_BYTES: "1G"
  ITERATIONS: "50"
  WORLD_SIZE: "4"
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf
spec:
  serviceName: nperf
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
        envFrom:
        - configMapRef:
            name: nperf-config
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
          [ "$RANK" != "0" ] && sleep 10

          /nperf/build/nperf \
            --nccl-bootstrap \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            --op $OPERATION \
            -b $MIN_BYTES -B $MAX_BYTES \
            -i $ITERATIONS
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Multi-GPU per Pod

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf-multigpu
spec:
  serviceName: nperf
  replicas: 2  # 2 nodes
  selector:
    matchLabels:
      app: nperf-multigpu
  template:
    metadata:
      labels:
        app: nperf-multigpu
    spec:
      containers:
      - name: nperf
        image: your-registry/nperf:latest
        env:
        - name: NCCL_COMM_ID
          value: "nperf-multigpu-0.nperf:5201"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        command: ["/bin/sh", "-c"]
        args:
        - |
          # Each pod has 4 GPUs
          RANK=$(echo $POD_NAME | grep -o '[0-9]*$')
          WORLD_SIZE=2
          GPUS_PER_POD=4

          [ "$RANK" != "0" ] && sleep 10

          /nperf/build/nperf \
            --nccl-bootstrap \
            --rank $RANK \
            --world-size $WORLD_SIZE \
            -n $GPUS_PER_POD \
            --op allreduce \
            -b 1M -B 1G
        resources:
          limits:
            nvidia.com/gpu: 4
```

## Node Affinity

Ensure pods run on different nodes:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf
spec:
  serviceName: nperf
  replicas: 4
  selector:
    matchLabels:
      app: nperf
  template:
    metadata:
      labels:
        app: nperf
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - nperf
            topologyKey: "kubernetes.io/hostname"
      containers:
      - name: nperf
        image: your-registry/nperf:latest
        # ... rest of container spec
```

## Network Policies

Allow NCCL traffic between pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nperf-network
spec:
  podSelector:
    matchLabels:
      app: nperf
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nperf
    ports:
    - protocol: TCP
      port: 5201
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: nperf
    ports:
    - protocol: TCP
      port: 5201
```

## Collecting Results

### With Persistent Volume

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nperf-results
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nperf
spec:
  # ... other spec
  template:
    spec:
      containers:
      - name: nperf
        image: your-registry/nperf:latest
        volumeMounts:
        - name: results
          mountPath: /results
        command: ["/bin/sh", "-c"]
        args:
        - |
          RANK=$(echo $POD_NAME | grep -o '[0-9]*$')

          # Only rank 0 saves results
          if [ "$RANK" = "0" ]; then
            OUTPUT="-J -o /results/benchmark_$(date +%Y%m%d_%H%M%S).json"
          fi

          /nperf/build/nperf \
            --nccl-bootstrap \
            --rank $RANK \
            --world-size 4 \
            --op allreduce \
            -b 1M -B 1G \
            $OUTPUT
      volumes:
      - name: results
        persistentVolumeClaim:
          claimName: nperf-results
```

## Debugging

### Check Pod Status

```bash
kubectl get pods -l app=nperf
kubectl describe pod nperf-0
```

### View Logs

```bash
kubectl logs nperf-0
kubectl logs nperf-1
```

### Interactive Debug

```bash
kubectl exec -it nperf-0 -- /bin/bash

# Inside pod
nvidia-smi
nc -zv nperf-1.nperf 5201
```

### Network Debug

```bash
# Test DNS resolution
kubectl exec nperf-1 -- nslookup nperf-0.nperf

# Test connectivity
kubectl exec nperf-1 -- nc -zv nperf-0.nperf 5201
```

## Cleanup

```bash
# Delete StatefulSet and service
kubectl delete statefulset nperf
kubectl delete service nperf

# Delete job
kubectl delete job nperf-benchmark

# Delete all nperf pods
kubectl delete pods -l app=nperf
```

## See Also

- [NCCL Bootstrap Mode](../coordination/nccl-bootstrap.md)
- [Docker Compose Example](../coordination/nccl-bootstrap.md#docker-compose-example)
- [Troubleshooting](../advanced/troubleshooting.md)
