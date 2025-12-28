#include "nperf/coordination/nccl_bootstrap_coordinator.h"
#include "nperf/log.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace nperf {

NcclBootstrapCoordinator::NcclBootstrapCoordinator() = default;

NcclBootstrapCoordinator::~NcclBootstrapCoordinator() {
    finalize();
}

void NcclBootstrapCoordinator::setRankInfo(int rank, int worldSize) {
    rank_ = rank;
    worldSize_ = worldSize;
}

bool NcclBootstrapCoordinator::parseCommId(std::string& host, int& port) {
    const char* commId = std::getenv("NCCL_COMM_ID");
    if (!commId || strlen(commId) == 0) {
        return false;
    }

    std::string s(commId);
    size_t colonPos = s.rfind(':');
    if (colonPos == std::string::npos) {
        return false;
    }

    host = s.substr(0, colonPos);
    if (host.empty()) {
        return false;
    }

    try {
        port = std::stoi(s.substr(colonPos + 1));
    } catch (...) {
        return false;
    }

    // Validate port range
    if (port < 1 || port > 65535) {
        return false;
    }

    return true;
}

void NcclBootstrapCoordinator::initialize(int /*argc*/, char** /*argv*/) {
    if (initialized_) {
        return;
    }

    if (rank_ < 0 || worldSize_ <= 0) {
        throw std::runtime_error("NcclBootstrapCoordinator: rank and worldSize must be set");
    }

    // Get hostname (ensure null-termination for edge case)
    char hostname[256];
    hostname[sizeof(hostname) - 1] = '\0';
    if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
        hostname_ = hostname;
    } else {
        hostname_ = "unknown";
    }

    // Parse NCCL_COMM_ID
    std::string bootstrapHost;
    int bootstrapPort;
    if (!parseCommId(bootstrapHost, bootstrapPort)) {
        throw std::runtime_error(
            "NCCL_COMM_ID environment variable must be set to 'host:port' for bootstrap mode");
    }

    logInfo("NCCL bootstrap mode: rank " + std::to_string(rank_) + "/" +
            std::to_string(worldSize_) + " using " + bootstrapHost + ":" +
            std::to_string(bootstrapPort));

    if (rank_ == 0) {
        // Rank 0 acts as the bootstrap server
        logInfo("Rank 0: Starting bootstrap server on port " + std::to_string(bootstrapPort));

        serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket_ < 0) {
            throw std::runtime_error("Failed to create bootstrap server socket: " +
                                    std::string(strerror(errno)));
        }

        int opt = 1;
        setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        setsockopt(serverSocket_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(bootstrapPort);

        if (bind(serverSocket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            int savedErrno = errno;
            close(serverSocket_);
            serverSocket_ = -1;
            throw std::runtime_error("Failed to bind bootstrap server to port " +
                                    std::to_string(bootstrapPort) + ": " +
                                    std::string(strerror(savedErrno)));
        }

        if (listen(serverSocket_, worldSize_ - 1) < 0) {
            int savedErrno = errno;
            close(serverSocket_);
            serverSocket_ = -1;
            throw std::runtime_error("Failed to listen on bootstrap server socket: " +
                                    std::string(strerror(savedErrno)));
        }

        // Accept connections from all other ranks
        // Initialize to -1 to detect missing connections
        clientSockets_.resize(worldSize_ - 1, -1);
        for (int i = 0; i < worldSize_ - 1; i++) {
            logInfo("Waiting for client " + std::to_string(i + 1) + " of " +
                   std::to_string(worldSize_ - 1) + "...");

            int clientSock = accept(serverSocket_, nullptr, nullptr);
            if (clientSock < 0) {
                finalize();
                throw std::runtime_error("Failed to accept connection: " +
                                        std::string(strerror(errno)));
            }

            // Receive the rank from the client
            int clientRank;
            if (!recvAll(clientSock, &clientRank, sizeof(clientRank))) {
                close(clientSock);
                finalize();
                throw std::runtime_error("Failed to receive rank from client");
            }

            if (clientRank < 1 || clientRank >= worldSize_) {
                close(clientSock);
                finalize();
                throw std::runtime_error("Invalid client rank: " + std::to_string(clientRank));
            }

            clientSockets_[clientRank - 1] = clientSock;
            logInfo("Rank " + std::to_string(clientRank) + " connected");
        }

        // Verify all ranks connected
        for (int i = 0; i < worldSize_ - 1; i++) {
            if (clientSockets_[i] < 0) {
                finalize();
                throw std::runtime_error("Missing connection from rank " +
                                        std::to_string(i + 1));
            }
        }

        logInfo("All " + std::to_string(worldSize_ - 1) + " ranks connected");
    } else {
        // Other ranks connect to rank 0
        logInfo("Connecting to bootstrap server at " + bootstrapHost + ":" +
               std::to_string(bootstrapPort));

        struct addrinfo hints, *result;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        int err = getaddrinfo(bootstrapHost.c_str(), std::to_string(bootstrapPort).c_str(),
                             &hints, &result);
        if (err != 0) {
            throw std::runtime_error("Failed to resolve bootstrap host: " +
                                    std::string(gai_strerror(err)));
        }

        connectionSocket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (connectionSocket_ < 0) {
            int savedErrno = errno;
            freeaddrinfo(result);
            throw std::runtime_error("Failed to create client socket: " +
                                    std::string(strerror(savedErrno)));
        }

        int opt = 1;
        setsockopt(connectionSocket_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

        // Set connection timeout
        struct timeval timeout;
        timeout.tv_sec = 30;
        timeout.tv_usec = 0;
        setsockopt(connectionSocket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(connectionSocket_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        if (connect(connectionSocket_, result->ai_addr, result->ai_addrlen) < 0) {
            int savedErrno = errno;
            close(connectionSocket_);
            connectionSocket_ = -1;
            freeaddrinfo(result);
            throw std::runtime_error("Failed to connect to bootstrap server: " +
                                    std::string(strerror(savedErrno)));
        }
        freeaddrinfo(result);

        // Send our rank
        if (!sendAll(connectionSocket_, &rank_, sizeof(rank_))) {
            finalize();
            throw std::runtime_error("Failed to send rank to bootstrap server");
        }

        logInfo("Connected to bootstrap server");
    }

    initialized_ = true;
}

void NcclBootstrapCoordinator::finalize() {
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

void NcclBootstrapCoordinator::barrier() {
    char buf = 1;

    if (rank_ == 0) {
        // Collect from all clients
        for (int sock : clientSockets_) {
            if (!recvAll(sock, &buf, 1)) {
                throw std::runtime_error("Barrier failed: failed to receive from rank");
            }
        }
        // Release all clients
        for (int sock : clientSockets_) {
            if (!sendAll(sock, &buf, 1)) {
                throw std::runtime_error("Barrier failed: failed to send to rank");
            }
        }
    } else {
        // Send to rank 0, wait for ack
        if (!sendAll(connectionSocket_, &buf, 1)) {
            throw std::runtime_error("Barrier failed: failed to send to rank 0");
        }
        if (!recvAll(connectionSocket_, &buf, 1)) {
            throw std::runtime_error("Barrier failed: failed to receive from rank 0");
        }
    }
}

void NcclBootstrapCoordinator::broadcastNcclId(ncclUniqueId* id, int root) {
    if (rank_ == root) {
        ncclGetUniqueId(id);
    }
    broadcast(id, sizeof(ncclUniqueId), root);
}

void NcclBootstrapCoordinator::broadcast(void* data, size_t size, int root) {
    if (rank_ == 0) {
        if (root == 0) {
            // We're root, send to all
            for (int sock : clientSockets_) {
                if (!sendAll(sock, data, size)) {
                    throw std::runtime_error("Broadcast failed: failed to send to rank");
                }
            }
        } else {
            // Root is another rank, relay
            if (root < 1 || root >= worldSize_) {
                throw std::runtime_error("Broadcast failed: invalid root rank");
            }
            // Receive from root
            if (!recvAll(clientSockets_[root - 1], data, size)) {
                throw std::runtime_error("Broadcast failed: failed to receive from root");
            }
            // Send to others
            for (int i = 0; i < static_cast<int>(clientSockets_.size()); i++) {
                if (i + 1 != root) {
                    if (!sendAll(clientSockets_[i], data, size)) {
                        throw std::runtime_error("Broadcast failed: failed to send to rank");
                    }
                }
            }
        }
    } else {
        if (rank_ == root) {
            // We're root, send to rank 0 for relay
            if (!sendAll(connectionSocket_, data, size)) {
                throw std::runtime_error("Broadcast failed: failed to send to rank 0");
            }
        } else {
            // Receive from rank 0
            if (!recvAll(connectionSocket_, data, size)) {
                throw std::runtime_error("Broadcast failed: failed to receive from rank 0");
            }
        }
    }
}

void NcclBootstrapCoordinator::allReduceSum(double* data, size_t count) {
    size_t bytes = count * sizeof(double);

    if (rank_ == 0) {
        std::vector<double> buffer(count);
        for (int sock : clientSockets_) {
            if (!recvAll(sock, buffer.data(), bytes)) {
                throw std::runtime_error("allReduceSum failed: receive error");
            }
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
        for (int sock : clientSockets_) {
            if (!sendAll(sock, data, bytes)) {
                throw std::runtime_error("allReduceSum failed: send error");
            }
        }
    } else {
        if (!sendAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: send error");
        }
        if (!recvAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: receive error");
        }
    }
}

void NcclBootstrapCoordinator::allReduceSum(int64_t* data, size_t count) {
    size_t bytes = count * sizeof(int64_t);

    if (rank_ == 0) {
        std::vector<int64_t> buffer(count);
        for (int sock : clientSockets_) {
            if (!recvAll(sock, buffer.data(), bytes)) {
                throw std::runtime_error("allReduceSum failed: receive error");
            }
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
        for (int sock : clientSockets_) {
            if (!sendAll(sock, data, bytes)) {
                throw std::runtime_error("allReduceSum failed: send error");
            }
        }
    } else {
        if (!sendAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: send error");
        }
        if (!recvAll(connectionSocket_, data, bytes)) {
            throw std::runtime_error("allReduceSum failed: receive error");
        }
    }
}

void NcclBootstrapCoordinator::gather(const void* send, void* recv, size_t size, int root) {
    // Gather is done via rank 0 which relays to root if needed
    size_t totalSize = worldSize_ * size;

    if (rank_ == 0) {
        // Rank 0 always collects from all clients
        std::vector<char> buffer(totalSize);

        // Copy rank 0's data
        std::memcpy(buffer.data(), send, size);

        // Receive from all clients
        for (size_t i = 0; i < clientSockets_.size(); i++) {
            char* ptr = buffer.data() + (i + 1) * size;
            if (!recvAll(clientSockets_[i], ptr, size)) {
                throw std::runtime_error("gather failed: receive error");
            }
        }

        if (root == 0) {
            // We're the root, copy to output
            std::memcpy(recv, buffer.data(), totalSize);
        } else {
            // Send gathered data to root
            if (!sendAll(clientSockets_[root - 1], buffer.data(), totalSize)) {
                throw std::runtime_error("gather failed: failed to send to root");
            }
        }
    } else if (rank_ == root) {
        // We're root but not rank 0
        // First send our data to rank 0
        if (!sendAll(connectionSocket_, send, size)) {
            throw std::runtime_error("gather failed: send error");
        }
        // Then receive the complete gathered buffer from rank 0
        if (!recvAll(connectionSocket_, recv, totalSize)) {
            throw std::runtime_error("gather failed: receive error from rank 0");
        }
    } else {
        // Not rank 0 and not root - just send to rank 0
        if (!sendAll(connectionSocket_, send, size)) {
            throw std::runtime_error("gather failed: send error");
        }
    }
}

bool NcclBootstrapCoordinator::sendAll(int socket, const void* data, size_t size) {
    const char* ptr = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t sent = send(socket, ptr, remaining, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;  // Retry on signal interrupt
            return false;
        }
        if (sent == 0) return false;  // Connection closed
        ptr += sent;
        remaining -= sent;
    }

    return true;
}

bool NcclBootstrapCoordinator::recvAll(int socket, void* data, size_t size) {
    char* ptr = static_cast<char*>(data);
    size_t remaining = size;

    while (remaining > 0) {
        ssize_t received = recv(socket, ptr, remaining, 0);
        if (received < 0) {
            if (errno == EINTR) continue;  // Retry on signal interrupt
            return false;
        }
        if (received == 0) return false;  // Connection closed
        ptr += received;
        remaining -= received;
    }

    return true;
}

} // namespace nperf
