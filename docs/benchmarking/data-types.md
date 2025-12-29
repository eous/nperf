# Data Types

nperf supports all NCCL data types for benchmarking. This guide explains each type, its use cases, and considerations.

## Supported Data Types

| Type | Size | Description | NCCL Type |
|------|------|-------------|-----------|
| `float32` | 4 bytes | 32-bit floating point (default) | `ncclFloat32` |
| `float64` | 8 bytes | 64-bit floating point | `ncclFloat64` |
| `float16` | 2 bytes | 16-bit floating point | `ncclFloat16` |
| `bfloat16` | 2 bytes | Brain floating point | `ncclBfloat16` |
| `int8` | 1 byte | 8-bit signed integer | `ncclInt8` |
| `uint8` | 1 byte | 8-bit unsigned integer | `ncclUint8` |
| `int32` | 4 bytes | 32-bit signed integer | `ncclInt32` |
| `uint32` | 4 bytes | 32-bit unsigned integer | `ncclUint32` |
| `int64` | 8 bytes | 64-bit signed integer | `ncclInt64` |
| `uint64` | 8 bytes | 64-bit unsigned integer | `ncclUint64` |

## Usage

Specify the data type with `--dtype`:

```bash
./nperf --op allreduce --dtype float32 -b 1M
./nperf --op allreduce --dtype float16 -b 1M
./nperf --op allreduce --dtype bfloat16 -b 1M
```

## Floating Point Types

### float32 (Default)

- **Size**: 4 bytes (32 bits)
- **Precision**: ~7 decimal digits
- **Range**: ±3.4 × 10³⁸
- **Use Case**: Standard deep learning training, general compute

```bash
./nperf --op allreduce --dtype float32 -b 1M -B 1G
```

### float64

- **Size**: 8 bytes (64 bits)
- **Precision**: ~15 decimal digits
- **Range**: ±1.8 × 10³⁰⁸
- **Use Case**: Scientific computing, high-precision requirements

```bash
./nperf --op allreduce --dtype float64 -b 1M -B 1G
```

### float16 (Half Precision)

- **Size**: 2 bytes (16 bits)
- **Precision**: ~3 decimal digits
- **Range**: ±6.5 × 10⁴
- **Use Case**: Mixed-precision training, inference acceleration

```bash
./nperf --op allreduce --dtype float16 -b 1M -B 1G
```

### bfloat16 (Brain Float)

- **Size**: 2 bytes (16 bits)
- **Precision**: ~2 decimal digits
- **Range**: Same as float32 (±3.4 × 10³⁸)
- **Use Case**: Deep learning with float32 dynamic range

```bash
./nperf --op allreduce --dtype bfloat16 -b 1M -B 1G
```

**Note**: bfloat16 has the same exponent range as float32, making it ideal for gradient synchronization where large dynamic range matters more than precision.

## Integer Types

### int8 / uint8

- **Size**: 1 byte (8 bits)
- **Range**: -128 to 127 (int8) or 0 to 255 (uint8)
- **Use Case**: Quantized models, int8 inference

```bash
./nperf --op allreduce --dtype int8 -b 1M -B 1G
./nperf --op allreduce --dtype uint8 -b 1M -B 1G
```

### int32 / uint32

- **Size**: 4 bytes (32 bits)
- **Range**: -2³¹ to 2³¹-1 (int32) or 0 to 2³²-1 (uint32)
- **Use Case**: Index operations, counting

```bash
./nperf --op allreduce --dtype int32 -b 1M -B 1G
./nperf --op allreduce --dtype uint32 -b 1M -B 1G
```

### int64 / uint64

- **Size**: 8 bytes (64 bits)
- **Range**: -2⁶³ to 2⁶³-1 (int64) or 0 to 2⁶⁴-1 (uint64)
- **Use Case**: Large counts, timestamps

```bash
./nperf --op allreduce --dtype int64 -b 1M -B 1G
./nperf --op allreduce --dtype uint64 -b 1M -B 1G
```

## Performance Considerations

### Bandwidth vs Element Count

With the same byte count, smaller data types process more elements:

| Dtype | Size | Elements in 1 MB |
|-------|------|------------------|
| float64 | 8 bytes | 131,072 |
| float32 | 4 bytes | 262,144 |
| float16 | 2 bytes | 524,288 |
| int8 | 1 byte | 1,048,576 |

### Effective Bandwidth

Raw bandwidth is the same regardless of data type. What changes is:

1. **Computation overhead**: Reduction operations may vary
2. **Memory access patterns**: Smaller types may have different cache behavior
3. **Hardware support**: Some GPUs optimize for specific types

### Mixed Precision Workflows

For realistic mixed-precision training benchmarks:

```bash
# Forward pass (fp16)
./nperf --op allreduce --dtype float16 -b 1M -B 1G

# Backward pass (fp32 master weights)
./nperf --op allreduce --dtype float32 -b 1M -B 1G
```

## Comparison Benchmarks

### Compare All Types at Fixed Size

```bash
for dtype in float32 float64 float16 bfloat16 int8 int32 int64; do
    echo "=== $dtype ==="
    ./nperf --op allreduce --dtype $dtype -b 256M -i 100
done
```

### JSON Output for Analysis

```bash
./nperf --op allreduce --dtype float32 -b 1M -B 1G -J -o float32.json
./nperf --op allreduce --dtype float16 -b 1M -B 1G -J -o float16.json
./nperf --op allreduce --dtype bfloat16 -b 1M -B 1G -J -o bfloat16.json
```

## Verification Considerations

When using `--verify`, tolerance settings depend on data type:

```bash
# Higher tolerance for lower precision
./nperf --op allreduce --dtype float16 --verify --verify-tolerance 1e-3

# Lower tolerance for higher precision
./nperf --op allreduce --dtype float64 --verify --verify-tolerance 1e-10
```

## Common Use Cases

| Scenario | Recommended Type |
|----------|-----------------|
| Standard DL training | float32 |
| Mixed-precision training | float16 or bfloat16 |
| Scientific computing | float64 |
| Quantized inference | int8 |
| Embedding tables | int64 |
| General benchmarking | float32 (default) |

## See Also

- [Collective Operations](collective-operations.md)
- [Verification](verification.md)
- [CLI Reference](../cli-reference.md)
