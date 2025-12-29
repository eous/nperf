# Multi-Node Socket Examples

Examples for running nperf across multiple nodes using TCP sockets (no MPI required).

## Prerequisites

- Network connectivity between nodes
- Same port accessible on all nodes (default: 5201)
- nperf binary on all nodes
- CUDA and NCCL installed on all nodes

## Basic Examples

### Two Nodes

```bash
# Node 1 (server, rank 0)
./nperf -s -n 1 --op allreduce -b 1M

# Node 2 (client, rank 1)
./nperf -c node1 --op allreduce -b 1M
```

### Three Nodes

```bash
# Node 1 (server, expects 2 clients)
./nperf -s -n 2 --op allreduce -b 1M -B 1G

# Node 2 (client)
./nperf -c node1 --op allreduce -b 1M -B 1G

# Node 3 (client)
./nperf -c node1 --op allreduce -b 1M -B 1G
```

### Four Nodes

```bash
# Node 1 (server, expects 3 clients)
./nperf -s -n 3 --op allreduce -b 1K -B 1G

# Nodes 2, 3, 4 (clients)
./nperf -c node1 --op allreduce -b 1K -B 1G
```

## Custom Port

```bash
# Server on port 6000
./nperf -s -n 2 -p 6000 --op allreduce -b 1M

# Clients connect to port 6000
./nperf -c server.example.com -p 6000 --op allreduce -b 1M
```

## Automation Scripts

### Simple Two-Node Script

```bash
#!/bin/bash
# run_two_node.sh

SERVER="node1"
CLIENT="node2"
PORT=5201

# Start server in background
ssh $SERVER "./nperf -s -n 1 -p $PORT --op allreduce -b 1M -B 1G" &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Start client
ssh $CLIENT "./nperf -c $SERVER -p $PORT --op allreduce -b 1M -B 1G"

# Wait for server to complete
wait $SERVER_PID
```

### Multi-Client Script

```bash
#!/bin/bash
# run_multi_node.sh

SERVER="node1"
CLIENTS=("node2" "node3" "node4")
PORT=5201
NUM_CLIENTS=${#CLIENTS[@]}

# Start server
echo "Starting server on $SERVER (expecting $NUM_CLIENTS clients)..."
ssh $SERVER "./nperf -s -n $NUM_CLIENTS -p $PORT --op allreduce -b 1M -B 1G -J" &
SERVER_PID=$!

# Wait for server
sleep 3

# Start clients in parallel
echo "Starting clients..."
for client in "${CLIENTS[@]}"; do
    ssh $client "./nperf -c $SERVER -p $PORT --op allreduce -b 1M -B 1G" &
done

# Wait for all
wait
echo "Benchmark complete"
```

### Parallel SSH (pdsh) Script

```bash
#!/bin/bash
# run_pdsh.sh

SERVER="node1"
PORT=5201

# Start server on first node
ssh $SERVER "./nperf -s -n 3 -p $PORT --op allreduce -b 1M -B 1G" &
sleep 3

# Start clients on other nodes
pdsh -w node[2-4] "./nperf -c $SERVER -p $PORT --op allreduce -b 1M -B 1G"

wait
```

## Cloud VM Examples

### AWS EC2

```bash
# On GPU instance 1 (server)
./nperf -s -n 1 -p 5201 --op allreduce -b 1M -B 1G

# On GPU instance 2 (client)
./nperf -c 10.0.0.10 -p 5201 --op allreduce -b 1M -B 1G
```

### GCP Compute Engine

```bash
# On VM 1 (server)
./nperf -s -n 2 -p 5201 --op allreduce -b 1M

# On VMs 2 and 3 (clients)
./nperf -c gpu-vm-1.us-central1-a.c.project.internal -p 5201 --op allreduce -b 1M
```

### Azure VMs

```bash
# On VM 1 (server)
./nperf -s -n 1 -p 5201 --op allreduce -b 1M

# On VM 2 (client)
./nperf -c 10.0.1.4 -p 5201 --op allreduce -b 1M
```

## Network Configuration

### Firewall Setup

```bash
# Allow port 5201 (iptables)
sudo iptables -A INPUT -p tcp --dport 5201 -j ACCEPT

# Allow port 5201 (firewalld)
sudo firewall-cmd --add-port=5201/tcp --permanent
sudo firewall-cmd --reload

# Allow port 5201 (ufw)
sudo ufw allow 5201/tcp
```

### Verify Connectivity

```bash
# From client, check server is reachable
nc -zv server 5201

# Check port is listening on server
netstat -tlnp | grep 5201
ss -tlnp | grep 5201
```

### Select Network Interface

```bash
# Set interface for NCCL
export NCCL_SOCKET_IFNAME=eth0

# Server
./nperf -s -n 1 --op allreduce -b 1M

# Client
./nperf -c 192.168.1.100 --op allreduce -b 1M
```

## With JSON Output

### Server Collects Results

```bash
# Server saves JSON results
./nperf -s -n 2 -p 5201 --op allreduce -b 1K -B 1G -J -o results.json

# Clients (no output needed)
./nperf -c server -p 5201 --op allreduce -b 1K -B 1G
```

## Complete Benchmark Scripts

### Health Check

```bash
#!/bin/bash
# socket_health_check.sh

SERVER=$1
NUM_CLIENTS=$2
PORT=${3:-5201}

if [ -z "$SERVER" ] || [ -z "$NUM_CLIENTS" ]; then
    echo "Usage: $0 <server_host> <num_clients> [port]"
    exit 1
fi

echo "Testing socket mode with $NUM_CLIENTS clients..."
echo "Server: $SERVER:$PORT"

# Quick connectivity test
./nperf -s -n $NUM_CLIENTS -p $PORT --op allreduce -b 1K -i 5

echo "Health check complete"
```

### Automated Multi-Node

```bash
#!/bin/bash
# socket_benchmark.sh

# Configuration
SERVER="gpu-server-1"
CLIENTS=("gpu-client-1" "gpu-client-2" "gpu-client-3")
PORT=5201
OUTPUT_DIR="socket_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

NUM_CLIENTS=${#CLIENTS[@]}

echo "Socket Benchmark"
echo "================"
echo "Server: $SERVER"
echo "Clients: ${CLIENTS[*]}"
echo "Output: $OUTPUT_DIR"
echo

# Operations to test
OPERATIONS="allreduce allgather broadcast reduce"

for op in $OPERATIONS; do
    echo "Testing $op..."

    # Start server
    ssh $SERVER "cd /path/to/nperf && ./nperf -s -n $NUM_CLIENTS -p $PORT --op $op -b 1K -B 1G -i 50 -J" > "$OUTPUT_DIR/${op}.json" &
    SERVER_PID=$!

    sleep 3

    # Start clients
    for client in "${CLIENTS[@]}"; do
        ssh $client "cd /path/to/nperf && ./nperf -c $SERVER -p $PORT --op $op -b 1K -B 1G -i 50" &
    done

    wait

    echo "  Saved to $OUTPUT_DIR/${op}.json"
done

echo
echo "Benchmark complete. Results in $OUTPUT_DIR/"
```

### Cloud Deployment Script

```bash
#!/bin/bash
# cloud_benchmark.sh

# Configuration (adjust for your cloud)
SERVER_IP="10.0.0.10"
CLIENT_IPS=("10.0.0.11" "10.0.0.12")
SSH_KEY="~/.ssh/cloud_key"
USER="ubuntu"
NPERF_PATH="/home/ubuntu/nperf"
PORT=5201

NUM_CLIENTS=${#CLIENT_IPS[@]}

echo "Starting server on $SERVER_IP..."
ssh -i $SSH_KEY $USER@$SERVER_IP \
    "cd $NPERF_PATH && ./nperf -s -n $NUM_CLIENTS -p $PORT --op allreduce -b 1M -B 1G -J" &
SERVER_PID=$!

sleep 5

echo "Starting $NUM_CLIENTS clients..."
for ip in "${CLIENT_IPS[@]}"; do
    ssh -i $SSH_KEY $USER@$ip \
        "cd $NPERF_PATH && ./nperf -c $SERVER_IP -p $PORT --op allreduce -b 1M -B 1G" &
done

wait $SERVER_PID
echo "Benchmark complete"
```

## Troubleshooting

### Connection Refused

```bash
# Check server is running
ssh server "netstat -tlnp | grep 5201"

# Check firewall
ssh server "iptables -L -n | grep 5201"

# Test with nc
nc -zv server 5201
```

### Connection Timeout

```bash
# Check network route
traceroute server

# Check if port is blocked
telnet server 5201

# Try alternate port
./nperf -s -n 1 -p 6000 ...
./nperf -c server -p 6000 ...
```

### Clients Not All Connecting

```bash
# Increase server startup delay
sleep 5  # Instead of sleep 3

# Ensure all clients have same options
# Check network from each client
for client in node2 node3 node4; do
    ssh $client "nc -zv server 5201"
done
```

### Port Already in Use

```bash
# Find process using port
lsof -i :5201
ss -tlnp | grep 5201

# Kill if needed
fuser -k 5201/tcp

# Or use different port
./nperf -s -n 1 -p 5202 ...
```

## Best Practices

1. **Start server first**: Always start server before clients
2. **Consistent options**: Use same benchmark options on all nodes
3. **Wait for startup**: Add delay after server start
4. **Check connectivity**: Test with nc before benchmark
5. **Use scripts**: Automate for reproducibility

## See Also

- [Socket Mode](../coordination/socket.md)
- [Performance Tuning](../advanced/performance-tuning.md)
- [Troubleshooting](../advanced/troubleshooting.md)
