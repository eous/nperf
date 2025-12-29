# Troubleshooting

This guide covers common issues and solutions when using nperf.

## Quick Diagnostics

### First Steps

```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version

# Check nperf help
./nperf --help

# Check topology
./nperf --topology
```

### Enable Debug Output

```bash
# NCCL debug
export NCCL_DEBUG=INFO
./nperf --op allreduce -b 1M

# Verbose nperf output
./nperf --op allreduce -b 1M -v --debug
```

## GPU Issues

### No GPUs Detected

**Symptom**: "No CUDA devices found" or similar error

**Diagnosis**:
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
ls -la /dev/nvidia*
```

**Solutions**:
```bash
# Unset restrictive CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Check driver
nvidia-smi -q | head -20

# Restart driver (requires root)
sudo nvidia-smi -pm 1
```

### GPU Memory Errors

**Symptom**: CUDA out of memory errors

**Diagnosis**:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**Solutions**:
```bash
# Kill processes using GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill

# Use smaller message sizes
./nperf --op allreduce -b 1M  # Instead of 1G

# Use fewer GPUs
./nperf -n 2 --op allreduce -b 256M
```

### GPU in Exclusive Mode

**Symptom**: Cannot access GPU, exclusive process mode

**Diagnosis**:
```bash
nvidia-smi -q | grep "Compute Mode"
```

**Solutions**:
```bash
# Change to default mode (requires root)
sudo nvidia-smi -c 0

# Or use the specific GPU that's available
CUDA_VISIBLE_DEVICES=1 ./nperf --op allreduce -b 1M
```

## NCCL Issues

### NCCL Initialization Timeout

**Symptom**: Hangs during NCCL communicator creation

**Diagnosis**:
```bash
export NCCL_DEBUG=INFO
./nperf --op allreduce -b 1M
```

**Solutions**:
```bash
# Increase timeout
export NCCL_TIMEOUT=600

# Check network interface
export NCCL_SOCKET_IFNAME=eth0

# Disable InfiniBand if not available
export NCCL_IB_DISABLE=1
```

### NCCL Version Mismatch

**Symptom**: NCCL library errors, version conflicts

**Diagnosis**:
```bash
# Check installed NCCL
ls /usr/lib/x86_64-linux-gnu/libnccl*
ldconfig -p | grep nccl
```

**Solutions**:
```bash
# Set library path explicitly
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Rebuild nperf with correct NCCL
cmake -DNCCL_ROOT=/path/to/nccl ..
make clean && make
```

### P2P Access Denied

**Symptom**: Slow performance, P2P disabled messages

**Diagnosis**:
```bash
export NCCL_DEBUG=INFO
./nperf --op allreduce -b 1M 2>&1 | grep -i p2p
```

**Solutions**:
```bash
# Check P2P support
nvidia-smi topo -p2p r

# Enable P2P in NCCL
export NCCL_P2P_LEVEL=5

# Or accept PCIe path
export NCCL_P2P_DISABLE=0
```

## Network Issues (Multi-Node)

### Connection Refused

**Symptom**: Cannot connect to server in socket mode

**Diagnosis**:
```bash
# Check server is running
netstat -tlnp | grep 5201

# Check connectivity
nc -zv server_host 5201
```

**Solutions**:
```bash
# Ensure server starts first
./nperf -s -n 2 --op allreduce -b 1M &
sleep 2
./nperf -c server_host --op allreduce -b 1M

# Check firewall
iptables -L -n | grep 5201

# Use different port
./nperf -s -n 2 -p 6000 --op allreduce -b 1M
./nperf -c server_host -p 6000 --op allreduce -b 1M
```

### Connection Timeout

**Symptom**: Client times out connecting to server

**Diagnosis**:
```bash
ping server_host
traceroute server_host
nc -zv -w5 server_host 5201
```

**Solutions**:
```bash
# Check routing
ip route get server_ip

# Use correct interface
export NCCL_SOCKET_IFNAME=eth0

# Verify hostname resolution
nslookup server_host
```

### InfiniBand Issues

**Symptom**: IB errors, slow multi-node performance

**Diagnosis**:
```bash
ibstat
ibv_devinfo
ib_write_bw -d mlx5_0
```

**Solutions**:
```bash
# Check IB device status
ibstatus

# Select specific HCA
export NCCL_IB_HCA=mlx5_0

# Check GID for RoCE
ibv_devinfo -v | grep GID

# Set correct GID index
export NCCL_IB_GID_INDEX=3
```

## MPI Issues

### MPI Initialization Fails

**Symptom**: MPI_Init errors

**Diagnosis**:
```bash
mpirun --version
mpirun -np 2 hostname
```

**Solutions**:
```bash
# Check MPI configuration
ompi_info | grep "Open MPI"

# Verify SSH access
ssh node1 hostname
ssh node2 hostname

# Check hostfile
cat hostfile
```

### Rank Mismatch

**Symptom**: Wrong number of ranks, ranks don't match GPUs

**Diagnosis**:
```bash
mpirun -np 4 bash -c 'echo "Rank: $OMPI_COMM_WORLD_RANK GPU: $CUDA_VISIBLE_DEVICES"'
```

**Solutions**:
```bash
# Explicit GPU binding
mpirun -np 4 --map-by ppr:1:gpu ./nperf --mpi --op allreduce -b 1M

# SLURM GPU binding
#SBATCH --gpus-per-task=1
srun ./nperf --mpi --op allreduce -b 1M
```

## Performance Issues

### Lower Than Expected Bandwidth

**Symptom**: Results significantly below hardware capability

**Diagnosis**:
```bash
# Check topology
./nperf --topology

# Check NVLink status
nvidia-smi nvlink -s

# Check IB status
ibstat
```

**Solutions**:
```bash
# Ensure NVLink is used
export NCCL_P2P_LEVEL=5

# Enable GPUDirect RDMA
export NCCL_NET_GDR_LEVEL=5

# Try different algorithms
./nperf --op allreduce --algo ring -b 1G
./nperf --op allreduce --algo tree -b 1G
```

### High Variance in Results

**Symptom**: Inconsistent timing between runs

**Solutions**:
```bash
# More warmup iterations
./nperf --op allreduce -b 1M -w 20

# More iterations for averaging
./nperf --op allreduce -b 1M -i 100

# Use CUDA graphs for consistent timing
./nperf --op allreduce --graph -b 1M -i 1000

# Check for background GPU activity
nvidia-smi dmon -s u
```

### CPU Bottleneck

**Symptom**: Small message latency is high

**Diagnosis**:
```bash
# Compare with/without CUDA graphs
./nperf --op allreduce -b 1K -i 1000
./nperf --op allreduce --graph -b 1K -i 1000
```

**Solutions**:
```bash
# Use CUDA graphs
./nperf --op allreduce --graph -b 1K -i 1000

# CPU affinity (MPI)
mpirun --bind-to core ./nperf --mpi --op allreduce -b 1K
```

## Verification Failures

### Floating-Point Errors

**Symptom**: Verification fails with small differences

**Solutions**:
```bash
# Increase tolerance for low precision types
./nperf --op allreduce --dtype float16 --verify --verify-tolerance 1e-2

# Check data type
./nperf --op allreduce --dtype float64 --verify --verify-tolerance 1e-10
```

### Hardware Errors

**Symptom**: Random verification failures, ECC errors

**Diagnosis**:
```bash
nvidia-smi -q | grep -i ecc
cuda-memcheck ./nperf --op allreduce --verify -b 1M
```

**Solutions**:
```bash
# Check GPU health
nvidia-smi -q | grep "Errors"

# Run GPU diagnostics
dcgmi diag -r 3
```

## Build Issues

### CMake Configuration Fails

**Symptom**: CMake cannot find CUDA or NCCL

**Solutions**:
```bash
# Specify paths
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DNCCL_ROOT=/usr/local/nccl \
      ..

# Check environment
echo $CUDA_HOME
echo $NCCL_ROOT
```

### Link Errors

**Symptom**: Undefined symbols during linking

**Solutions**:
```bash
# Rebuild clean
rm -rf build && mkdir build && cd build
cmake ..
make -j

# Check library paths
ldd ./nperf | grep -E "(nccl|cuda)"
```

## Log Collection

For reporting issues, collect:

```bash
#!/bin/bash
# collect_diagnostics.sh

echo "=== System Info ===" > diag.txt
uname -a >> diag.txt
cat /etc/os-release >> diag.txt

echo "=== GPU Info ===" >> diag.txt
nvidia-smi >> diag.txt

echo "=== CUDA Version ===" >> diag.txt
nvcc --version >> diag.txt

echo "=== NCCL Debug ===" >> diag.txt
NCCL_DEBUG=INFO ./nperf --op allreduce -b 1M 2>> diag.txt

echo "=== Topology ===" >> diag.txt
./nperf --topology >> diag.txt

echo "Diagnostics saved to diag.txt"
```

## See Also

- [Environment Variables](environment-variables.md)
- [Performance Tuning](performance-tuning.md)
- [Coordination Overview](../coordination/overview.md)
