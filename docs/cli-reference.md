# CLI Reference

Complete reference for all nperf command-line options.

## Usage

```bash
nperf [OPTIONS]
```

---

## Coordination Mode Options

These options are mutually exclusive. Default is `--local`.

| Option | Description |
|--------|-------------|
| `--local` | Single-node mode (default) |
| `--mpi` | Use MPI for multi-node coordination |
| `-s, --server` | Run as socket server (multi-node) |
| `-c, --client HOST` | Connect to socket server at HOST |
| `--nccl-bootstrap` | Use NCCL native bootstrap mode |

### Mode-Specific Options

| Option | Mode | Default | Description |
|--------|------|---------|-------------|
| `-n, --num-gpus N` | Local/Server | All available | Number of GPUs or expected clients |
| `-p, --port PORT` | Socket | 5201 | TCP port for socket coordination |
| `--rank N` | NCCL Bootstrap | Required | Rank ID (0 to world-size-1) |
| `--world-size N` | NCCL Bootstrap | Required | Total number of ranks |

---

## Benchmark Configuration

### Message Size

| Option | Default | Description |
|--------|---------|-------------|
| `-b, --bytes SIZE` | 1K | Minimum message size |
| `-B, --max-bytes SIZE` | Same as -b | Maximum message size |
| `-S, --step FACTOR` | 2 | Multiplicative step factor |

**Size format**: Number with optional suffix: `K` (1024), `M` (1024²), `G` (1024³), `T` (1024⁴)

Examples:
- `1024` - 1024 bytes
- `1K` - 1 KB (1024 bytes)
- `16M` - 16 MB
- `1G` - 1 GB

### Duration

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --iters N` | 20 | Iterations per message size |
| `-w, --warmup N` | 5 | Warmup iterations (not timed) |
| `-O, --omit SECONDS` | 0 | Omit first N seconds from results |
| `-t, --time SECONDS` | - | Time-based mode (overrides -i) |

### Operation Selection

| Option | Default | Description |
|--------|---------|-------------|
| `--op OPERATION` | allreduce | Collective operation |
| `--dtype TYPE` | float32 | Data type |
| `--redop OP` | sum | Reduction operation |
| `--root N` | 0 | Root rank for rooted operations |

**Operations**: `allreduce`, `allgather`, `broadcast`, `reduce`, `reducescatter`, `alltoall`, `gather`, `scatter`, `sendrecv`

**Data Types**: `float32`, `float64`, `float16`, `bfloat16`, `int32`, `int64`, `int8`, `uint8`, `uint32`, `uint64`

**Reduction Ops**: `sum`, `prod`, `min`, `max`, `avg`

---

## NCCL Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--algo ALGO` | auto | NCCL algorithm selection |
| `--proto PROTO` | auto | NCCL protocol selection |

**Algorithms**: `auto`, `ring`, `tree`, `collnetdirect`, `collnetchain`, `nvls`

**Protocols**: `auto`, `simple`, `ll`, `ll128`

---

## CUDA Options

| Option | Default | Description |
|--------|---------|-------------|
| `--graph` | Disabled | Enable CUDA Graph capture |
| `--device N` | -1 | CUDA device ID (-1 = auto) |

---

## Verification

| Option | Default | Description |
|--------|---------|-------------|
| `--verify` | Disabled | Enable per-iteration verification |
| `--verify-tolerance TOL` | 1e-5 | Floating-point tolerance |

---

## Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `-J, --json` | Disabled | Output in JSON format |
| `-o, --output FILE` | stdout | Write results to file |
| `-v, --verbose` | Disabled | Verbose logging |
| `--debug` | Disabled | Enable NCCL debug output |

---

## Topology Options

| Option | Default | Description |
|--------|---------|-------------|
| `--topology` | Disabled | Show topology only, skip benchmark |
| `--topo-format FMT` | matrix | Topology output format |
| `--show-transport` | Disabled | Show transport information |

**Topology Formats**: `matrix`, `tree`, `dot`, `json`

---

## Miscellaneous

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `--version` | Show version information |
| `--interval SECONDS` | Progress report interval (default: 1.0) |

---

## Default Values Summary

| Setting | Default |
|---------|---------|
| Coordination Mode | Local |
| Operation | AllReduce |
| Data Type | float32 |
| Reduction Op | sum |
| Min Bytes | 1K |
| Max Bytes | Same as min |
| Step Factor | 2 |
| Iterations | 20 |
| Warmup | 5 |
| Algorithm | auto |
| Protocol | auto |
| Port | 5201 |
| Output Format | Text |
| Device | -1 (auto) |
| Verify Tolerance | 1e-5 |

---

## Examples

### Basic Benchmark
```bash
./nperf --op allreduce -b 1K -B 1G
```

### Multi-GPU with Specific Count
```bash
./nperf -n 4 --op allreduce -b 1M -i 100
```

### MPI Multi-Node
```bash
mpirun -np 8 ./nperf --mpi --op allreduce -b 1M -B 1G
```

### Socket Multi-Node
```bash
# Server (expects 3 clients)
./nperf -s -n 3 --op allreduce -b 1M

# Clients
./nperf -c server-hostname --op allreduce -b 1M
```

### NCCL Bootstrap Mode
```bash
export NCCL_COMM_ID=node0:5201
./nperf --nccl-bootstrap --rank 0 --world-size 4 --op allreduce
```

### JSON Output
```bash
./nperf --op allreduce -b 1M -B 1G -J -o results.json
```

### With Verification
```bash
./nperf --op allreduce -b 1M --verify --verify-tolerance 1e-6
```

### Force Algorithm and Protocol
```bash
./nperf --op allreduce --algo ring --proto ll -b 1M
```

### CUDA Graph Mode
```bash
./nperf --op allreduce --graph -b 1M -i 1000
```

### Topology Only
```bash
./nperf --topology --topo-format dot > topology.dot
dot -Tpng topology.dot -o topology.png
```
