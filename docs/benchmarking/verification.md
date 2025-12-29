# Verification

nperf can verify the correctness of NCCL operations to ensure data integrity. This is useful for debugging, validation, and catching hardware or software issues.

## Overview

Verification computes expected results and compares them against actual NCCL output.

## Verification Modes

| Mode | When Checked | Performance Impact |
|------|--------------|-------------------|
| None (default) | Never | None |
| Per-Iteration | After each iteration | High |

## Enabling Verification

### Basic Verification

```bash
./nperf --op allreduce --verify -b 1M
```

### Custom Tolerance

```bash
./nperf --op allreduce --verify --verify-tolerance 1e-6 -b 1M
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--verify` | Disabled | Enable per-iteration verification |
| `--verify-tolerance TOL` | 1e-5 | Floating-point comparison tolerance |

## How Verification Works

### AllReduce Verification

1. Initialize buffer with known pattern (e.g., rank-based values)
2. Execute NCCL AllReduce
3. Compute expected result: sum of all rank values
4. Compare actual vs expected within tolerance

```
Input (rank i):     buffer[j] = i + 1
Expected output:    buffer[j] = sum(1..N) = N*(N+1)/2
```

### Broadcast Verification

1. Root rank has source data
2. Execute NCCL Broadcast
3. All ranks compare received data with source

### Other Operations

Each operation has specific verification logic:
- **Reduce**: Root compares against computed reduction
- **AllGather**: Each rank verifies received segments
- **ReduceScatter**: Each rank verifies its portion
- **Gather/Scatter**: Data placement verified

## Tolerance Guidelines

Choose tolerance based on data type and operation:

| Data Type | Recommended Tolerance |
|-----------|-----------------------|
| float64 | 1e-10 to 1e-12 |
| float32 | 1e-5 to 1e-6 |
| float16 | 1e-2 to 1e-3 |
| bfloat16 | 1e-2 to 1e-3 |
| int* | 0 (exact match) |

```bash
# High precision
./nperf --op allreduce --dtype float64 --verify --verify-tolerance 1e-10

# Standard precision
./nperf --op allreduce --dtype float32 --verify --verify-tolerance 1e-5

# Low precision
./nperf --op allreduce --dtype float16 --verify --verify-tolerance 1e-2
```

## Performance Impact

Verification adds overhead:
- Buffer copies for comparison
- CPU-side computation of expected values
- Synchronization points

**Recommendations**:
- Use verification for debugging, not production benchmarks
- Run fewer iterations with verification
- Use smaller message sizes for quick validation

```bash
# Quick verification run
./nperf --op allreduce --verify -b 1K -B 1M -i 10

# Production benchmark (no verification)
./nperf --op allreduce -b 1K -B 1G -i 100
```

## Error Output

When verification fails:

```
Verification FAILED at iteration 5
  Position: 1024
  Expected: 120.000000
  Actual:   120.000122
  Difference: 0.000122 (tolerance: 0.00001)
```

## Debugging with Verification

### Step 1: Isolate the Issue

```bash
# Start with single GPU
./nperf -n 1 --op allreduce --verify -b 1K

# Add GPUs progressively
./nperf -n 2 --op allreduce --verify -b 1K
./nperf -n 4 --op allreduce --verify -b 1K
```

### Step 2: Vary Message Size

```bash
# Small messages
./nperf --op allreduce --verify -b 1K -i 10

# Large messages
./nperf --op allreduce --verify -b 1G -i 5
```

### Step 3: Test Different Operations

```bash
for op in allreduce broadcast reduce allgather; do
    echo "Testing $op..."
    ./nperf --op $op --verify -b 1M -i 10
done
```

### Step 4: Enable Debug Output

```bash
./nperf --op allreduce --verify --debug -b 1M
```

## Common Verification Failures

### Tolerance Too Tight

**Symptom**: Failures with very small differences

**Solution**: Increase tolerance for the data type

```bash
# Too tight for float32
./nperf --dtype float32 --verify --verify-tolerance 1e-10  # May fail

# Appropriate for float32
./nperf --dtype float32 --verify --verify-tolerance 1e-5   # Should pass
```

### Accumulation Errors

**Symptom**: Larger errors with more GPUs or larger messages

**Cause**: Floating-point accumulation order differs

**Solution**: Use larger tolerance or exact reduction algorithms

### Hardware Issues

**Symptom**: Random, inconsistent failures

**Cause**: Memory errors, interconnect issues

**Diagnosis**:
```bash
# Run GPU memory test
nvidia-smi -pm 1
cuda-memcheck ./nperf --op allreduce --verify -b 1M

# Check ECC errors
nvidia-smi -q | grep -i ecc
```

### NVLink Issues

**Symptom**: Failures only with NVLink-connected GPUs

**Diagnosis**:
```bash
# Check NVLink status
nvidia-smi nvlink -s

# Force PCIe path
export NCCL_P2P_DISABLE=1
./nperf --op allreduce --verify -b 1M
```

## Verification with Different Reduction Operations

```bash
# Sum (default)
./nperf --op allreduce --redop sum --verify -b 1M

# Product
./nperf --op allreduce --redop prod --verify -b 1M

# Min/Max
./nperf --op allreduce --redop min --verify -b 1M
./nperf --op allreduce --redop max --verify -b 1M
```

## Best Practices

1. **Always verify new setups**: Run verification when setting up new clusters
2. **Periodic validation**: Include verification in health checks
3. **Match production types**: Verify with actual data types used
4. **Document tolerance**: Record expected tolerances for your hardware

## Example: Full Validation Suite

```bash
#!/bin/bash
# validate_nccl.sh

SIZES="1K 1M 256M"
OPS="allreduce allgather broadcast reduce"
DTYPES="float32 float16 bfloat16"

for op in $OPS; do
    for dtype in $DTYPES; do
        for size in $SIZES; do
            echo "Testing: $op $dtype $size"
            if ! ./nperf --op $op --dtype $dtype --verify -b $size -i 5; then
                echo "FAILED: $op $dtype $size"
                exit 1
            fi
        done
    done
done

echo "All verification tests passed!"
```

## See Also

- [Collective Operations](collective-operations.md)
- [Data Types](data-types.md)
- [Troubleshooting](../advanced/troubleshooting.md)
