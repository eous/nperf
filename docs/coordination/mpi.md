# MPI Mode

MPI mode uses the Message Passing Interface for multi-node coordination. It's ideal for HPC clusters with existing MPI infrastructure.

## Overview

- **Use Case**: Multi-node HPC clusters
- **Dependencies**: MPI library (OpenMPI, MPICH, Intel MPI)
- **Rank**: From MPI (`MPI_Comm_rank`)
- **World Size**: From MPI (`MPI_Comm_size`)

## Prerequisites

1. MPI library installed and configured
2. nperf built with MPI support (`-DNPERF_BUILD_MPI=ON`)
3. Passwordless SSH between nodes (for most MPI launchers)
4. Shared filesystem or identical binary on all nodes

### Check MPI Support
```bash
./nperf --help | grep mpi
# Should show: --mpi  Use MPI coordination
```

## CLI Options

```bash
mpirun -np N ./nperf --mpi [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--mpi` | Enable MPI coordination |

All standard MPI launcher options apply (hostfile, ranks per node, etc.).

## Examples

### Basic Multi-Node
```bash
mpirun -np 8 ./nperf --mpi --op allreduce -b 1M -B 1G
```

### With Hostfile
```bash
# hosts.txt:
# node1 slots=4
# node2 slots=4

mpirun -np 8 --hostfile hosts.txt ./nperf --mpi --op allreduce -b 1M
```

### SLURM Integration
```bash
srun -n 8 ./nperf --mpi --op allreduce -b 1M
```

### PBS/Torque Integration
```bash
mpirun ./nperf --mpi --op allreduce -b 1M
```

## Job Scheduler Examples

### SLURM Job Script
```bash
#!/bin/bash
#SBATCH --job-name=nperf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00

srun ./nperf --mpi --op allreduce -b 1K -B 1G -o results_${SLURM_JOB_ID}.json -J
```

### PBS Job Script
```bash
#!/bin/bash
#PBS -N nperf
#PBS -l nodes=2:ppn=4:gpus=4
#PBS -l walltime=00:30:00

cd $PBS_O_WORKDIR
mpirun ./nperf --mpi --op allreduce -b 1K -B 1G
```

### LSF Job Script
```bash
#!/bin/bash
#BSUB -J nperf
#BSUB -n 8
#BSUB -R "span[ptile=4]"
#BSUB -gpu "num=4"

mpirun ./nperf --mpi --op allreduce -b 1K -B 1G
```

## How It Works

1. MPI launcher starts N processes across nodes
2. Each process calls `MPI_Init_thread()` with `MPI_THREAD_FUNNELED`
3. Rank 0 generates NCCL unique ID
4. ID is broadcast via `MPI_Bcast()`
5. All ranks initialize NCCL communicator
6. Benchmarks run with MPI barriers for synchronization
7. Results aggregated via `MPI_Allreduce()` and `MPI_Gather()`

## Environment Variables

MPI sets these automatically:
- `OMPI_COMM_WORLD_RANK` (OpenMPI)
- `PMI_RANK` (SLURM)
- `MPI_LOCALRANKID` (various)

nperf respects CUDA device assignment from job schedulers.

## GPU Assignment

### Automatic (Recommended)
Most MPI launchers with GPU support automatically set `CUDA_VISIBLE_DEVICES`:

```bash
srun --gpus-per-task=1 ./nperf --mpi --op allreduce
```

### Manual GPU Binding
```bash
# OpenMPI with GPU binding
mpirun -np 8 --map-by ppr:4:node:pe=1 \
  --bind-to core \
  ./nperf --mpi --op allreduce -b 1M
```

### SLURM GPU Binding
```bash
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
srun ./nperf --mpi --op allreduce
```

## Performance Tips

1. **Use GPU-aware MPI** if available (not required by nperf, but good practice)
2. **Match ranks to GPUs**: One MPI rank per GPU for best results
3. **Enable NCCL topology detection**: Set `NCCL_TOPO_FILE` if needed
4. **Network optimization**:
   ```bash
   export NCCL_IB_DISABLE=0  # Enable InfiniBand
   export NCCL_NET_GDR_LEVEL=2  # Enable GPUDirect RDMA
   ```

## Troubleshooting

### MPI Initialization Fails
```bash
# Check MPI is working
mpirun -np 2 hostname

# Check OpenMPI version
mpirun --version
```

### NCCL Timeout
```bash
# Increase timeout
export NCCL_TIMEOUT=1800  # 30 minutes

# Enable debug
./nperf --mpi --debug --op allreduce -b 1M
```

### GPU Not Found
```bash
# Verify GPU visibility per rank
mpirun -np 2 bash -c 'echo "Rank $OMPI_COMM_WORLD_RANK: $(nvidia-smi -L)"'
```

### Network Issues
```bash
# Check InfiniBand
ibstat

# Force TCP
export NCCL_IB_DISABLE=1
mpirun -np 8 ./nperf --mpi --op allreduce -b 1M
```

## When to Use MPI Mode

**Recommended for:**
- HPC clusters with existing MPI
- SLURM/PBS/LSF job schedulers
- Production benchmarking
- Established MPI infrastructure

**Consider alternatives when:**
- No MPI installed → Use [Socket Mode](socket.md)
- Kubernetes deployment → Use [NCCL Bootstrap Mode](nccl-bootstrap.md)
- Single node → Use [Local Mode](local.md)

## See Also

- [Coordination Overview](overview.md)
- [Multi-Node MPI Examples](../examples/multi-node-mpi.md)
- [Troubleshooting](../advanced/troubleshooting.md)
