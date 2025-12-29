# Collective Operations

nperf supports all major NCCL collective operations. This guide explains each operation, its use cases, and how to benchmark them.

## Overview

| Operation | Description | Rooted | Reduction |
|-----------|-------------|--------|-----------|
| AllReduce | Reduce + broadcast to all | No | Yes |
| AllGather | Gather data from all to all | No | No |
| Broadcast | Send from root to all | Yes | No |
| Reduce | Reduce to root only | Yes | Yes |
| ReduceScatter | Reduce + scatter results | No | Yes |
| AlltoAll | All-to-all personalized exchange | No | No |
| Gather | Gather to root | Yes | No |
| Scatter | Scatter from root | Yes | No |
| SendRecv | Point-to-point communication | N/A | No |

## AllReduce

**Description**: Combines values from all processes using a reduction operation, then distributes the result to all processes.

**Use Case**: Gradient synchronization in distributed deep learning.

```
Before:                 After:
Rank 0: [1,2,3]        Rank 0: [6,9,12]
Rank 1: [2,3,4]   →    Rank 1: [6,9,12]
Rank 2: [3,4,5]        Rank 2: [6,9,12]
```

**CLI Example**:
```bash
./nperf --op allreduce -b 1M -B 1G --redop sum
```

**Bus Bandwidth Factor**: `2 * (n-1) / n` where n is world size

This accounts for the fact that AllReduce is typically implemented as ReduceScatter + AllGather.

## AllGather

**Description**: Gathers data from all processes and distributes the combined data to all processes.

**Use Case**: Collecting partial results for inference, weight sharing.

```
Before:                 After:
Rank 0: [A]            Rank 0: [A,B,C]
Rank 1: [B]       →    Rank 1: [A,B,C]
Rank 2: [C]            Rank 2: [A,B,C]
```

**CLI Example**:
```bash
./nperf --op allgather -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## Broadcast

**Description**: Sends data from one root process to all other processes.

**Use Case**: Distributing model weights, hyperparameters, initial state.

```
Before:                 After:
Rank 0: [A,B,C]        Rank 0: [A,B,C]
Rank 1: [?,?,?]   →    Rank 1: [A,B,C]  (root=0)
Rank 2: [?,?,?]        Rank 2: [A,B,C]
```

**CLI Example**:
```bash
./nperf --op broadcast --root 0 -b 1M -B 1G
```

**Bus Bandwidth Factor**: `1.0`

## Reduce

**Description**: Reduces values from all processes to a single root process.

**Use Case**: Collecting metrics, aggregating statistics.

```
Before:                 After:
Rank 0: [1,2,3]        Rank 0: [6,9,12]
Rank 1: [2,3,4]   →    Rank 1: [2,3,4]   (root=0, sum)
Rank 2: [3,4,5]        Rank 2: [3,4,5]
```

**CLI Example**:
```bash
./nperf --op reduce --root 0 --redop sum -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## ReduceScatter

**Description**: Reduces data across all processes and scatters the result, with each process receiving a portion.

**Use Case**: Distributed computation where each process needs different parts of the result.

```
Before:                 After:
Rank 0: [1,2,3]        Rank 0: [6]      (element 0 reduced)
Rank 1: [2,3,4]   →    Rank 1: [9]      (element 1 reduced)
Rank 2: [3,4,5]        Rank 2: [12]     (element 2 reduced)
```

**CLI Example**:
```bash
./nperf --op reducescatter --redop sum -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## AlltoAll

**Description**: Each process sends distinct data to every other process.

**Use Case**: Transposing distributed matrices, FFT communication patterns.

```
Before:                 After:
Rank 0: [A0,A1,A2]     Rank 0: [A0,B0,C0]
Rank 1: [B0,B1,B2]  →  Rank 1: [A1,B1,C1]
Rank 2: [C0,C1,C2]     Rank 2: [A2,B2,C2]
```

**CLI Example**:
```bash
./nperf --op alltoall -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## Gather

**Description**: Collects data from all processes to a single root process.

**Use Case**: Collecting distributed results for final processing.

```
Before:                 After:
Rank 0: [A]            Rank 0: [A,B,C]
Rank 1: [B]       →    Rank 1: [B]       (root=0)
Rank 2: [C]            Rank 2: [C]
```

**CLI Example**:
```bash
./nperf --op gather --root 0 -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## Scatter

**Description**: Distributes data from root process to all processes.

**Use Case**: Distributing work chunks, data partitioning.

```
Before:                 After:
Rank 0: [A,B,C]        Rank 0: [A]
Rank 1: [?]       →    Rank 1: [B]       (root=0)
Rank 2: [?]            Rank 2: [C]
```

**CLI Example**:
```bash
./nperf --op scatter --root 0 -b 1M -B 1G
```

**Bus Bandwidth Factor**: `(n-1) / n` where n is world size

## SendRecv

**Description**: Point-to-point communication between pairs of processes.

**Use Case**: Pipelined communication, neighbor exchange patterns.

```
Rank 0 sends to Rank 1
Rank 1 receives from Rank 0
```

**CLI Example**:
```bash
./nperf --op sendrecv -b 1M -B 1G
```

**Bus Bandwidth Factor**: `1.0`

## Reduction Operations

For operations that perform reductions (AllReduce, Reduce, ReduceScatter), specify the reduction operation:

| Operation | Description | Example |
|-----------|-------------|---------|
| `sum` | Add all values (default) | 1+2+3 = 6 |
| `prod` | Multiply all values | 1×2×3 = 6 |
| `min` | Minimum value | min(1,2,3) = 1 |
| `max` | Maximum value | max(1,2,3) = 3 |
| `avg` | Average value | avg(1,2,3) = 2 |

**CLI Example**:
```bash
./nperf --op allreduce --redop max -b 1M
```

## Performance Characteristics

### Bandwidth Scaling

| Operation | Small Messages | Large Messages |
|-----------|----------------|----------------|
| AllReduce | Latency-bound | Ring optimal |
| AllGather | Latency-bound | High bandwidth |
| Broadcast | Tree optimal | Bandwidth limited |
| Reduce | Tree optimal | Bandwidth limited |

### Algorithm Selection

NCCL automatically selects the best algorithm, but you can override:

```bash
# Force ring algorithm
./nperf --op allreduce --algo ring -b 1M -B 1G

# Force tree algorithm
./nperf --op allreduce --algo tree -b 1M -B 1G
```

## Benchmarking All Operations

Compare all operations at a specific size:

```bash
for op in allreduce allgather broadcast reduce reducescatter alltoall gather scatter; do
    echo "=== $op ==="
    ./nperf --op $op -b 1G -i 100
done
```

## See Also

- [Data Types](data-types.md)
- [Performance Tuning](../advanced/performance-tuning.md)
- [CLI Reference](../cli-reference.md)
