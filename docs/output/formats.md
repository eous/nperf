# Output Formats

nperf supports multiple output formats for different use cases. This guide explains each format and how to use them.

## Available Formats

| Format | Option | Best For |
|--------|--------|----------|
| Text | (default) | Human reading, quick checks |
| JSON | `-J, --json` | Programmatic analysis, logging |

## Text Format (Default)

Human-readable tabular output.

### Example

```bash
./nperf --op allreduce -b 1K -B 1G
```

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
  1K            12.34         0.08               0.06
  2K            12.56         0.16               0.12
  4K            12.89         0.31               0.23
  ...
  512M          45678.90      11.21              8.41
  1G            91234.56      10.97              8.23
```

### Columns

| Column | Description |
|--------|-------------|
| Size | Message size per GPU |
| Time (us) | Average latency in microseconds |
| Bandwidth (GB/s) | Raw bandwidth (Size / Time) |
| Bus BW (GB/s) | Bus bandwidth (accounts for collective algorithm) |

## JSON Format

Machine-readable structured output for analysis tools.

### Enable JSON

```bash
./nperf --op allreduce -b 1K -B 1G -J
```

Or:

```bash
./nperf --op allreduce -b 1K -B 1G --json
```

### JSON Structure

```json
{
  "version": "1.0.0",
  "config": {
    "operation": "allreduce",
    "dataType": "float32",
    "reduceOp": "sum",
    "numGpus": 4,
    "iterations": 20,
    "warmupIterations": 5,
    "algorithm": "auto",
    "protocol": "auto"
  },
  "results": [
    {
      "bytes": 1024,
      "iterations": 20,
      "timeUs": {
        "avg": 12.34,
        "min": 11.20,
        "max": 14.56,
        "stddev": 0.89,
        "p50": 12.15,
        "p95": 13.80,
        "p99": 14.32
      },
      "bandwidth": 0.083,
      "busBandwidth": 0.062
    },
    {
      "bytes": 2048,
      "iterations": 20,
      "timeUs": {
        "avg": 12.56,
        "min": 11.45,
        "max": 14.78,
        "stddev": 0.92,
        "p50": 12.40,
        "p95": 14.10,
        "p99": 14.55
      },
      "bandwidth": 0.163,
      "busBandwidth": 0.122
    }
  ],
  "topology": {
    "gpus": [
      {"id": 0, "name": "NVIDIA A100-SXM4-80GB", "uuid": "GPU-xxx"},
      {"id": 1, "name": "NVIDIA A100-SXM4-80GB", "uuid": "GPU-yyy"}
    ],
    "links": [
      {"from": 0, "to": 1, "type": "NVLink", "bandwidth": 600}
    ]
  },
  "environment": {
    "hostname": "gpu-node-01",
    "cudaVersion": "12.0",
    "ncclVersion": "2.18.1",
    "driverVersion": "525.105.17"
  }
}
```

### JSON Fields

#### config

| Field | Type | Description |
|-------|------|-------------|
| operation | string | Collective operation name |
| dataType | string | Data type used |
| reduceOp | string | Reduction operation (if applicable) |
| numGpus | integer | Number of GPUs |
| iterations | integer | Timed iterations per size |
| warmupIterations | integer | Warmup iterations |
| algorithm | string | NCCL algorithm |
| protocol | string | NCCL protocol |

#### results (per message size)

| Field | Type | Description |
|-------|------|-------------|
| bytes | integer | Message size in bytes |
| iterations | integer | Iterations completed |
| timeUs.avg | float | Average time (microseconds) |
| timeUs.min | float | Minimum time |
| timeUs.max | float | Maximum time |
| timeUs.stddev | float | Standard deviation |
| timeUs.p50 | float | 50th percentile (median) |
| timeUs.p95 | float | 95th percentile |
| timeUs.p99 | float | 99th percentile |
| bandwidth | float | Raw bandwidth (GB/s) |
| busBandwidth | float | Bus bandwidth (GB/s) |

## File Output

Write results to a file:

```bash
# Text output to file
./nperf --op allreduce -b 1M -o results.txt

# JSON output to file
./nperf --op allreduce -b 1M -J -o results.json
```

### Option

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output FILE` | stdout | Write results to FILE |

## Parsing JSON Output

### Python

```python
import json

with open('results.json') as f:
    data = json.load(f)

# Access results
for result in data['results']:
    print(f"Size: {result['bytes']}, BW: {result['bandwidth']:.2f} GB/s")

# Get configuration
print(f"Operation: {data['config']['operation']}")
print(f"GPUs: {data['config']['numGpus']}")
```

### jq (Command Line)

```bash
# Get all bandwidths
jq '.results[].bandwidth' results.json

# Get max bandwidth
jq '[.results[].bandwidth] | max' results.json

# Filter results > 1MB
jq '.results[] | select(.bytes >= 1048576)' results.json

# Pretty print configuration
jq '.config' results.json
```

### Bash Script

```bash
#!/bin/bash
# Parse JSON results

results_file="results.json"

# Get peak bandwidth using jq
peak_bw=$(jq '[.results[].bandwidth] | max' "$results_file")
echo "Peak Bandwidth: $peak_bw GB/s"

# Get number of GPUs
num_gpus=$(jq '.config.numGpus' "$results_file")
echo "GPUs: $num_gpus"
```

## Combining Multiple Runs

```bash
# Run benchmarks with different configurations
./nperf --op allreduce -b 1M -J -o allreduce.json
./nperf --op allgather -b 1M -J -o allgather.json
./nperf --op broadcast -b 1M -J -o broadcast.json

# Combine with jq
jq -s '.' allreduce.json allgather.json broadcast.json > combined.json
```

## Verbose Output

Enable verbose logging:

```bash
./nperf --op allreduce -b 1M -v
```

Verbose mode adds:
- Per-iteration timing
- Memory allocation details
- NCCL initialization info

## Debug Output

Enable NCCL debug output:

```bash
./nperf --op allreduce -b 1M --debug
```

Or via environment:

```bash
export NCCL_DEBUG=INFO
./nperf --op allreduce -b 1M
```

## Progress Reporting

For long-running benchmarks, progress is reported:

```bash
./nperf --op allreduce -b 1K -B 1G --interval 2.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--interval SECONDS` | 1.0 | Progress report interval |

## Example Workflows

### Quick Performance Check

```bash
./nperf --op allreduce -b 256M
```

### Detailed Analysis

```bash
./nperf --op allreduce -b 1K -B 1G -J -o detailed.json
python analyze_results.py detailed.json
```

### Automated Testing

```bash
#!/bin/bash
./nperf --op allreduce -b 1G -J -o results.json

# Check if bandwidth meets threshold
bw=$(jq '.results[0].bandwidth' results.json)
threshold=10.0

if (( $(echo "$bw < $threshold" | bc -l) )); then
    echo "FAIL: Bandwidth $bw < $threshold GB/s"
    exit 1
fi
echo "PASS: Bandwidth $bw >= $threshold GB/s"
```

## See Also

- [Topology Visualization](topology.md)
- [CLI Reference](../cli-reference.md)
- [Examples](../examples/single-node.md)
