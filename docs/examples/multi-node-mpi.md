# Multi-Node MPI Examples

Examples for running nperf across multiple nodes using MPI.

## Prerequisites

- MPI library (OpenMPI, MPICH, Intel MPI)
- nperf built with MPI support (`-DNPERF_BUILD_MPI=ON`)
- Passwordless SSH between nodes
- Shared filesystem or nperf binary on all nodes
- CUDA and NCCL installed on all nodes

## Check MPI Support

```bash
./nperf --help | grep mpi
# Should show: --mpi  Use MPI coordination
```

## Basic Examples

### Two Nodes, 2 GPUs Each

```bash
mpirun -np 4 -H node1:2,node2:2 ./nperf --mpi --op allreduce -b 1M
```

### Four Nodes, 4 GPUs Each

```bash
mpirun -np 16 -H node1:4,node2:4,node3:4,node4:4 ./nperf --mpi --op allreduce -b 1M -B 1G
```

### Using Hostfile

Create `hostfile.txt`:
```
node1 slots=4
node2 slots=4
node3 slots=4
node4 slots=4
```

Run:
```bash
mpirun -np 16 --hostfile hostfile.txt ./nperf --mpi --op allreduce -b 1M -B 1G
```

## SLURM Examples

### Basic SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=nperf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00

srun ./nperf --mpi --op allreduce -b 1K -B 1G
```

### With GPU Binding

```bash
#!/bin/bash
#SBATCH --job-name=nperf-bench
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=nperf_%j.out

# Load modules
module load cuda nccl

# Set NCCL environment
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Run benchmark
srun --gpus-per-task=1 ./nperf --mpi --op allreduce -b 1K -B 1G -J -o results_${SLURM_JOB_ID}.json
```

### Full Benchmark Suite (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=nperf-suite
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=nperf_suite_%j.out

module load cuda nccl

OUTPUT_DIR="results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Test each operation
for op in allreduce allgather broadcast reduce reducescatter; do
    echo "Testing $op..."
    srun ./nperf --mpi --op $op -b 1K -B 1G -i 50 -J -o "$OUTPUT_DIR/${op}.json"
done

echo "Results saved to $OUTPUT_DIR/"
```

## PBS/Torque Examples

### Basic PBS Job

```bash
#!/bin/bash
#PBS -N nperf
#PBS -l nodes=2:ppn=4:gpus=4
#PBS -l walltime=00:30:00
#PBS -j oe

cd $PBS_O_WORKDIR

mpirun ./nperf --mpi --op allreduce -b 1K -B 1G
```

### PBS with Environment Setup

```bash
#!/bin/bash
#PBS -N nperf-bench
#PBS -l nodes=4:ppn=8:gpus=8
#PBS -l walltime=01:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Load modules
module load cuda/12.0 nccl/2.18

# Environment
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Get node list
NODES=$(cat $PBS_NODEFILE | sort -u | tr '\n' ',' | sed 's/,$//')
NPROCS=$(wc -l < $PBS_NODEFILE)

mpirun -np $NPROCS ./nperf --mpi --op allreduce -b 1K -B 1G -J -o results.json
```

## LSF Examples

### Basic LSF Job

```bash
#!/bin/bash
#BSUB -J nperf
#BSUB -n 16
#BSUB -R "span[ptile=4]"
#BSUB -gpu "num=4"
#BSUB -W 00:30

mpirun ./nperf --mpi --op allreduce -b 1M -B 1G
```

## OpenMPI Specific

### GPU Binding with OpenMPI

```bash
mpirun -np 8 \
    --map-by ppr:4:node \
    --bind-to core \
    --mca btl_tcp_if_include eth0 \
    ./nperf --mpi --op allreduce -b 1G
```

### With InfiniBand

```bash
mpirun -np 16 \
    --hostfile hostfile.txt \
    --mca btl openib,self \
    --mca btl_openib_allow_ib true \
    ./nperf --mpi --op allreduce -b 1G
```

## MPICH Specific

### Basic MPICH

```bash
mpiexec -n 16 -f hostfile.txt ./nperf --mpi --op allreduce -b 1G
```

### With UCX

```bash
mpiexec -n 16 -f hostfile.txt \
    -env UCX_TLS rc,cuda_copy,cuda_ipc \
    ./nperf --mpi --op allreduce -b 1G
```

## Network Configuration

### Force TCP (No InfiniBand)

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

mpirun -np 8 --hostfile hostfile.txt ./nperf --mpi --op allreduce -b 1G
```

### Enable InfiniBand with GPUDirect

```bash
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0

mpirun -np 16 --hostfile hostfile.txt ./nperf --mpi --op allreduce -b 1G
```

### Select Network Interface

```bash
# For NCCL
export NCCL_SOCKET_IFNAME=ib0

# For MPI
mpirun --mca btl_tcp_if_include ib0 -np 8 ./nperf --mpi --op allreduce -b 1G
```

## Debugging

### Enable MPI Debug

```bash
mpirun -np 4 --display-map --display-binding ./nperf --mpi --op allreduce -b 1M
```

### Enable NCCL Debug

```bash
export NCCL_DEBUG=INFO
mpirun -np 8 ./nperf --mpi --op allreduce -b 1M
```

### Check GPU Assignment

```bash
mpirun -np 8 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK: $(nvidia-smi -L)"'
```

### Verify Connectivity

```bash
# Test MPI
mpirun -np 8 --hostfile hostfile.txt hostname

# Test NCCL with small message
mpirun -np 8 --hostfile hostfile.txt ./nperf --mpi --op allreduce -b 1K -i 5 -v
```

## Performance Optimization

### Optimal Process Placement

```bash
# One rank per GPU, 4 GPUs per node
mpirun -np 32 \
    --map-by ppr:4:node:PE=1 \
    --bind-to core \
    ./nperf --mpi --op allreduce -b 1G
```

### Affinity Settings

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=8
srun --cpu-bind=cores ./nperf --mpi --op allreduce -b 1G
```

## Complete Benchmark Scripts

### Multi-Node Health Check

```bash
#!/bin/bash
# mpi_health_check.sh

NODES=${1:-2}
GPUS_PER_NODE=${2:-4}
TOTAL=$((NODES * GPUS_PER_NODE))

echo "Testing $TOTAL GPUs across $NODES nodes..."

# Quick connectivity test
mpirun -np $TOTAL ./nperf --mpi --op allreduce -b 1K -i 5 -v

# Bandwidth test
mpirun -np $TOTAL ./nperf --mpi --op allreduce -b 1G -i 20
```

### Scaling Study

```bash
#!/bin/bash
# scaling_study.sh

echo "GPU Scaling Study"
echo "================="

for nodes in 1 2 4 8; do
    gpus=$((nodes * 4))
    echo "Nodes: $nodes, GPUs: $gpus"

    mpirun -np $gpus -H $(echo node{1..$nodes} | tr ' ' ',') \
        ./nperf --mpi --op allreduce -b 1G -i 50 -J | \
        jq -r '"  Bandwidth: \(.results[0].bandwidth) GB/s"'
done
```

### Full Multi-Node Sweep

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00

OUTPUT_DIR="mpi_results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Operations
for op in allreduce allgather broadcast reduce reducescatter; do
    echo "Testing $op..."
    srun ./nperf --mpi --op $op -b 1K -B 4G -i 50 -J -o "$OUTPUT_DIR/${op}.json"
done

# Data types
for dtype in float32 float16 bfloat16; do
    echo "Testing $dtype..."
    srun ./nperf --mpi --op allreduce --dtype $dtype -b 256M -i 100 -J -o "$OUTPUT_DIR/dtype_${dtype}.json"
done

echo "Results saved to $OUTPUT_DIR/"
```

## Troubleshooting

### MPI Rank Issues

```bash
# Debug rank assignment
mpirun -np 8 bash -c 'echo "Host: $(hostname), Rank: $OMPI_COMM_WORLD_RANK"'
```

### NCCL Timeout

```bash
export NCCL_TIMEOUT=600  # 10 minutes
mpirun -np 16 ./nperf --mpi --op allreduce -b 1G
```

### GPU Not Found

```bash
# Check per-rank GPU visibility
mpirun -np 8 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
```

## See Also

- [MPI Mode](../coordination/mpi.md)
- [Performance Tuning](../advanced/performance-tuning.md)
- [Environment Variables](../advanced/environment-variables.md)
- [Troubleshooting](../advanced/troubleshooting.md)
