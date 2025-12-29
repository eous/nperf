# Socket Mode

Socket mode uses TCP sockets for multi-node coordination without requiring MPI. It's ideal for cloud environments and quick multi-node testing.

## Overview

- **Use Case**: Multi-node without MPI
- **Dependencies**: None (pure TCP sockets)
- **Architecture**: Server-client model
- **Port**: Configurable (default: 5201)

## How It Works

```
┌─────────────────┐     TCP      ┌─────────────────┐
│  Server (Rank 0)│◄────────────►│  Client (Rank 1)│
│    ./nperf -s   │              │  ./nperf -c     │
└─────────────────┘              └─────────────────┘
         ▲                                ▲
         │            TCP                 │
         └────────────────────────────────┘
                      │
              ┌───────┴───────┐
              │ Client (Rank 2)│
              │  ./nperf -c    │
              └───────────────┘
```

1. **Server** starts first, listens on specified port
2. **Clients** connect to server
3. Server assigns ranks to clients (0 for server, 1+ for clients)
4. NCCL ID generated on server, broadcast to clients
5. All nodes initialize NCCL and run benchmark

## CLI Options

### Server Mode
```bash
./nperf -s [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --server` | - | Run as server (Rank 0) |
| `-n, --num-gpus N` | 1 | Expected number of clients |
| `-p, --port PORT` | 5201 | TCP port to listen on |

### Client Mode
```bash
./nperf -c HOST [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-c, --client HOST` | - | Server hostname/IP |
| `-p, --port PORT` | 5201 | Server port |

## Examples

### Two Nodes
```bash
# Node 1 (server, rank 0)
./nperf -s -n 1 --op allreduce -b 1M

# Node 2 (client, rank 1)
./nperf -c node1 --op allreduce -b 1M
```

### Four Nodes
```bash
# Node 1 (server, expects 3 clients)
./nperf -s -n 3 --op allreduce -b 1M -B 1G

# Nodes 2, 3, 4 (clients)
./nperf -c node1 --op allreduce -b 1M -B 1G
./nperf -c node1 --op allreduce -b 1M -B 1G
./nperf -c node1 --op allreduce -b 1M -B 1G
```

### Custom Port
```bash
# Server
./nperf -s -n 2 -p 6000 --op allreduce -b 1M

# Clients
./nperf -c server.example.com -p 6000 --op allreduce -b 1M
```

### With JSON Output
```bash
# Server
./nperf -s -n 1 --op allreduce -b 1K -B 1G -J -o results.json

# Client
./nperf -c server --op allreduce -b 1K -B 1G
```

## Startup Scripts

### Bash Script for Multiple Clients
```bash
#!/bin/bash
# run_benchmark.sh

SERVER="node1"
CLIENTS=("node2" "node3" "node4")
PORT=5201

# Start server in background
ssh $SERVER "./nperf -s -n ${#CLIENTS[@]} -p $PORT --op allreduce -b 1M" &
sleep 2  # Wait for server to start

# Start clients
for client in "${CLIENTS[@]}"; do
    ssh $client "./nperf -c $SERVER -p $PORT --op allreduce -b 1M" &
done

wait
```

### Parallel SSH (pdsh) Example
```bash
# Start server on first node
ssh node1 "./nperf -s -n 3 -p 5201 --op allreduce -b 1M" &
sleep 2

# Start clients on other nodes
pdsh -w node[2-4] "./nperf -c node1 -p 5201 --op allreduce -b 1M"
```

## Connection Sequence

1. Server binds to port and listens
2. Server waits for `N` client connections (30-second timeout per client)
3. Each client connects and receives its rank
4. Server broadcasts world size to all clients
5. Server generates and broadcasts NCCL unique ID
6. All nodes create NCCL communicator
7. Benchmark runs with TCP-based barriers
8. Results gathered at server

## Synchronization Protocol

### Barrier
- Server collects one byte from each client
- Server sends one byte to all clients (release)

### Broadcast
- If root=0: Server sends to all clients
- If root≠0: Client sends to server, server relays

### AllReduce
- All clients send values to server
- Server aggregates and sends result back

## Timeouts

All socket operations have a 30-second timeout:
- Server accept timeout
- Client connect timeout
- Send/receive timeout

## Firewall Configuration

Ensure the socket port is open:

```bash
# Check if port is accessible
nc -zv server 5201

# Open port (example for iptables)
iptables -A INPUT -p tcp --dport 5201 -j ACCEPT

# Open port (example for firewalld)
firewall-cmd --add-port=5201/tcp --permanent
firewall-cmd --reload
```

## Troubleshooting

### Connection Refused
```bash
# Verify server is running
netstat -tlnp | grep 5201

# Check firewall
telnet server 5201
```

### Connection Timeout
```bash
# Check network connectivity
ping server

# Check port accessibility
nc -zv server 5201

# Try alternate port
./nperf -s -p 6000 -n 1 ...
./nperf -c server -p 6000 ...
```

### Port Already in Use
```bash
# Find process using port
lsof -i :5201

# Use different port
./nperf -s -p 5202 -n 1 ...
```

### Clients Don't All Connect
- Verify all clients have same benchmark options
- Check network routes between nodes
- Ensure server started before clients
- Increase startup delay in scripts

## When to Use Socket Mode

**Recommended for:**
- Cloud VMs (AWS, GCP, Azure)
- Quick multi-node testing
- Environments without MPI
- Custom deployment scripts

**Consider alternatives when:**
- HPC cluster with MPI → Use [MPI Mode](mpi.md)
- Kubernetes → Use [NCCL Bootstrap Mode](nccl-bootstrap.md)
- Single node → Use [Local Mode](local.md)

## See Also

- [Coordination Overview](overview.md)
- [Multi-Node Socket Examples](../examples/multi-node-socket.md)
- [Troubleshooting](../advanced/troubleshooting.md)
