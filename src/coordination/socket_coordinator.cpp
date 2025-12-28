#include "nperf/coordination/socket_coordinator.h"
#include "nperf/log.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace nperf {

SocketCoordinator::SocketCoordinator() = default;

SocketCoordinator::~SocketCoordinator() {
    cleanup();
}

void SocketCoordinator::cleanup() {
    if (connectionSocket_ >= 0) {
        close(connectionSocket_);
        connectionSocket_ = -1;
    }

    for (int sock : clientSockets_) {
        if (sock >= 0) {
            close(sock);
        }
    }
    clientSockets_.clear();

    if (serverSocket_ >= 0) {
        close(serverSocket_);
        serverSocket_ = -1;
    }

    initialized_ = false;
}

void SocketCoordinator::setServerMode(int port, int expectedClients) {
    isServer_ = true;
    port_ = port;
    expectedClients_ = expectedClients;
}

void SocketCoordinator::setClientMode(const std::string& serverHost, int port) {
    isServer_ = false;
    serverHost_ = serverHost;
    port_ = port;
}

void SocketCoordinator::initialize(int /*argc*/, char** /*argv*/) {
    if (initialized_) {
        return;
    }

    // Get hostname
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname_ = hostname;
    } else {
        hostname_ = "unknown";
    }

    if (isServer_) {
        initializeServer(port_, expectedClients_);
    } else {
        initializeClient(serverHost_, port_);
    }

    initialized_ = true;
}

void SocketCoordinator::initializeServer(int port, int expectedClients) {
    logInfo("Starting socket server on port " + std::to_string(port) +
           " (expecting " + std::to_string(expectedClients) + " clients)");

    // Create server socket
    logDebug("Creating server socket...");
    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        throw std::runtime_error("Failed to create server socket");
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(serverSocket_);
        serverSocket_ = -1;
        throw std::runtime_error("Failed to set SO_REUSEADDR");
    }
    if (setsockopt(serverSocket_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
        close(serverSocket_);
        serverSocket_ = -1;
        throw std::runtime_error("Failed to set TCP_NODELAY");
    }

    // Set accept timeout (30 seconds)
    struct timeval timeout;
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;
    setsockopt(serverSocket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // Bind
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    logDebug("Binding to port " + std::to_string(port) + "...");
    if (bind(serverSocket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(serverSocket_);
        serverSocket_ = -1;
        throw std::runtime_error("Failed to bind server socket to port " +
                                std::to_string(port));
    }

    // Listen
    if (listen(serverSocket_, expectedClients) < 0) {
        close(serverSocket_);
        serverSocket_ = -1;
        throw std::runtime_error("Failed to listen on server socket");
    }

    logInfo("Server listening, waiting for clients...");

    // Accept connections from clients
    worldSize_ = expectedClients + 1;  // Clients + server
    rank_ = 0;  // Server is always rank 0

    clientSockets_.resize(expectedClients);
    for (int i = 0; i < expectedClients; i++) {
        logInfo("Waiting for client " + std::to_string(i + 1) + "/" +
               std::to_string(expectedClients) + "...");

        int clientSock = accept(serverSocket_, nullptr, nullptr);
        if (clientSock < 0) {
            cleanup();
            throw std::runtime_error("Failed to accept client connection");
        }

        // Receive client's rank (they send their requested rank)
        int clientRank;
        if (!recvAll(clientSock, &clientRank, sizeof(clientRank))) {
            close(clientSock);
            cleanup();
            throw std::runtime_error("Failed to receive client rank");
        }

        // Assign rank if client sent -1 (auto-assign)
        if (clientRank < 0) {
            clientRank = i + 1;
        }

        // Validate rank bounds
        if (clientRank < 1 || clientRank > expectedClients) {
            close(clientSock);
            cleanup();
            throw std::runtime_error("Invalid client rank: " + std::to_string(clientRank) +
                                    " (expected 1-" + std::to_string(expectedClients) + ")");
        }

        logInfo("Client connected, assigned rank " + std::to_string(clientRank));

        // Send back assigned rank and world size
        int info[2] = {clientRank, worldSize_};
        if (!sendAll(clientSock, info, sizeof(info))) {
            close(clientSock);
            cleanup();
            throw std::runtime_error("Failed to send rank info to client");
        }

        clientSockets_[clientRank - 1] = clientSock;
    }

    logInfo("All " + std::to_string(expectedClients) + " clients connected");
}

void SocketCoordinator::initializeClient(const std::string& serverHost, int port) {
    logInfo("Connecting to server at " + serverHost + ":" + std::to_string(port));

    // Resolve server address
    logDebug("Resolving server address...");
    struct addrinfo hints, *result;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int err = getaddrinfo(serverHost.c_str(), std::to_string(port).c_str(),
                         &hints, &result);
    if (err != 0) {
        throw std::runtime_error("Failed to resolve server address: " +
                                std::string(gai_strerror(err)));
    }

    // Create socket and connect
    logDebug("Creating client socket...");
    connectionSocket_ = socket(result->ai_family, result->ai_socktype,
                               result->ai_protocol);
    if (connectionSocket_ < 0) {
        freeaddrinfo(result);
        throw std::runtime_error("Failed to create client socket");
    }

    // Set TCP_NODELAY
    int opt = 1;
    if (setsockopt(connectionSocket_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
        close(connectionSocket_);
        connectionSocket_ = -1;
        freeaddrinfo(result);
        throw std::runtime_error("Failed to set TCP_NODELAY on client socket");
    }

    // Set connect/recv timeout (30 seconds)
    struct timeval timeout;
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;
    setsockopt(connectionSocket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(connectionSocket_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    logDebug("Connecting to server...");
    if (connect(connectionSocket_, result->ai_addr, result->ai_addrlen) < 0) {
        close(connectionSocket_);
        connectionSocket_ = -1;
        freeaddrinfo(result);
        throw std::runtime_error("Failed to connect to server");
    }

    freeaddrinfo(result);
    logInfo("Connected to server");

    // Send our requested rank (-1 for auto-assign)
    logDebug("Exchanging rank information...");
    int requestedRank = -1;
    if (!sendAll(connectionSocket_, &requestedRank, sizeof(requestedRank))) {
        cleanup();
        throw std::runtime_error("Failed to send rank request");
    }

    // Receive assigned rank and world size
    int info[2];
    if (!recvAll(connectionSocket_, info, sizeof(info))) {
        cleanup();
        throw std::runtime_error("Failed to receive rank info");
    }

    rank_ = info[0];
    worldSize_ = info[1];

    logInfo("Assigned rank " + std::to_string(rank_) + " of " + std::to_string(worldSize_));
}

void SocketCoordinator::finalize() {
    cleanup();
}

void SocketCoordinator::barrier() {
    // Simple barrier: all send 1 byte, then receive ack
    char buf = 1;

    if (isServer_) {
        // Collect from all clients
        for (int sock : clientSockets_) {
            if (!recvAll(sock, &buf, 1)) {
                throw std::runtime_error("Barrier failed: failed to receive from client");
            }
        }
        // Release all clients
        for (int sock : clientSockets_) {
            if (!sendAll(sock, &buf, 1)) {
                throw std::runtime_error("Barrier failed: failed to send to client");
            }
        }
    } else {
        // Send to server, wait for ack
        if (!sendAll(connectionSocket_, &buf, 1)) {
            throw std::runtime_error("Barrier failed: failed to send to server");
        }
        if (!recvAll(connectionSocket_, &buf, 1)) {
            throw std::runtime_error("Barrier failed: failed to receive from server");
        }
    }
}

void SocketCoordinator::broadcastNcclId(ncclUniqueId* id, int root) {
    if (rank_ == root) {
        // Generate ID if we're root
        ncclGetUniqueId(id);
    }

    broadcast(id, sizeof(ncclUniqueId), root);
}

void SocketCoordinator::broadcast(void* data, size_t size, int root) {
    if (isServer_) {
        if (root == 0) {
            // Server is root, broadcast to all clients
            for (int sock : clientSockets_) {
                if (!sendAll(sock, data, size)) {
                    throw std::runtime_error("Broadcast failed: failed to send to client");
                }
            }
        } else {
            // Root is a client, server needs to relay
            // Validate root index
            if (root < 1 || root > static_cast<int>(clientSockets_.size())) {
                throw std::runtime_error("Broadcast failed: invalid root rank " + std::to_string(root));
            }
            // Receive from root client
            if (!recvAll(clientSockets_[root - 1], data, size)) {
                throw std::runtime_error("Broadcast failed: failed to receive from root client");
            }
            // Send to other clients (not root)
            for (int i = 0; i < static_cast<int>(clientSockets_.size()); i++) {
                if (i + 1 != root) {
                    if (!sendAll(clientSockets_[i], data, size)) {
                        throw std::runtime_error("Broadcast failed: failed to send to client");
                    }
                }
            }
        }
    } else {
        // Client side
        if (rank_ == root) {
            // We are the root client, send to server for relay
            if (!sendAll(connectionSocket_, data, size)) {
                throw std::runtime_error("Broadcast failed: failed to send to server");
            }
        } else {
            // Receive from server
            if (!recvAll(connectionSocket_, data, size)) {
                throw std::runtime_error("Broadcast failed: failed to receive from server");
            }
        }
    }
}

void SocketCoordinator::allReduceSum(double* data, size_t count) {
    size_t bytes = count * sizeof(double);

    if (isServer_) {
        // Collect from all clients and sum
        std::vector<double> buffer(count);
        for (int sock : clientSockets_) {
            if (!recvAll(sock, buffer.data(), bytes)) {
                throw std::runtime_error("allReduceSum failed: failed to receive from client");
            }
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
        // Send result back to all clients
        for (int sock : clientSockets_) {
            if (!sendAll(sock, data, bytes)) {
                throw std::runtime_error("allReduceSum failed: failed to send to client");
            }
        }
    } else {
        // Send to server
        if (!sendAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: failed to send to server");
        }
        // Receive result
        if (!recvAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: failed to receive from server");
        }
    }
}

void SocketCoordinator::allReduceSum(int64_t* data, size_t count) {
    size_t bytes = count * sizeof(int64_t);

    if (isServer_) {
        std::vector<int64_t> buffer(count);
        for (int sock : clientSockets_) {
            if (!recvAll(sock, buffer.data(), bytes)) {
                throw std::runtime_error("allReduceSum failed: failed to receive from client");
            }
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
        for (int sock : clientSockets_) {
            if (!sendAll(sock, data, bytes)) {
                throw std::runtime_error("allReduceSum failed: failed to send to client");
            }
        }
    } else {
        if (!sendAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: failed to send to server");
        }
        if (!recvAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: failed to receive from server");
        }
    }
}

void SocketCoordinator::gather(const void* send, void* recv, size_t size, int root) {
    if (rank_ == root) {
        // Copy own data
        std::memcpy(static_cast<char*>(recv) + rank_ * size, send, size);

        if (isServer_) {
            // Receive from clients
            for (size_t i = 0; i < clientSockets_.size(); i++) {
                char* ptr = static_cast<char*>(recv) + (i + 1) * size;
                if (!recvAll(clientSockets_[i], ptr, size)) {
                    throw std::runtime_error("gather failed: failed to receive from client");
                }
            }
        }
    } else {
        // Send to root
        if (!isServer_) {
            if (!sendAll(connectionSocket_, send, size)) {
                throw std::runtime_error("gather failed: failed to send to server");
            }
        }
    }
}

bool SocketCoordinator::sendAll(int socket, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t sent = send(socket, ptr, remaining, 0);
        if (sent <= 0) {
            return false;
        }
        ptr += sent;
        remaining -= sent;
    }

    return true;
}

bool SocketCoordinator::recvAll(int socket, void* data, size_t size) {
    char* ptr = static_cast<char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t received = recv(socket, ptr, remaining, 0);
        if (received <= 0) {
            return false;
        }
        ptr += received;
        remaining -= received;
    }

    return true;
}

} // namespace nperf
