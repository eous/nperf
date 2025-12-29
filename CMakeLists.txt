cmake_minimum_required(VERSION 3.18)
project(nperf VERSION 1.0.0 LANGUAGES CXX CUDA)

# Options
option(NPERF_BUILD_MPI "Build with MPI support" ON)
option(NPERF_BUILD_TESTS "Build tests" ON)
option(NPERF_BUILD_EXAMPLES "Build examples" ON)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find required dependencies
find_package(CUDAToolkit REQUIRED)

# Find NCCL
find_package(NCCL REQUIRED)

# Find NVML (part of CUDA toolkit)
find_package(NVML REQUIRED)

# Optional MPI
if(NPERF_BUILD_MPI)
    find_package(MPI)
    if(MPI_FOUND)
        message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
        message(WARNING "MPI not found, building without MPI support")
        set(NPERF_BUILD_MPI OFF)
    endif()
endif()

# Fetch dependencies
include(FetchContent)

# nlohmann/json for JSON output (header-only library)
# We use EXCLUDE_FROM_ALL to prevent it from being part of install/export
set(JSON_Install OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    # Don't add_subdirectory - just use the include path directly
endif()
set(NLOHMANN_JSON_INCLUDE_DIR "${json_SOURCE_DIR}/include")

# CLI11 for argument parsing (don't install with nperf)
set(CLI11_INSTALL OFF CACHE INTERNAL "")
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.1
)
FetchContent_MakeAvailable(cli11)

# Version header
configure_file(
    "${CMAKE_SOURCE_DIR}/include/nperf/version.h.in"
    "${CMAKE_BINARY_DIR}/include/nperf/version.h"
    @ONLY
)

# Compiler warnings
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
)

# Core library (static) - no coordination dependencies
add_library(nperf_core STATIC
    src/core/timing.cpp
    src/core/memory.cpp
    src/core/collective.cpp
    src/core/metrics.cpp
    src/core/graph.cpp
    src/topology/detector.cpp
    src/topology/nvlink.cpp
    src/topology/pcie.cpp
    src/topology/rdma.cpp
    src/output/formatter.cpp
    src/output/json_formatter.cpp
    src/output/text_formatter.cpp
    src/output/topo_visualizer.cpp
    src/verification/verifier.cpp
)

target_include_directories(nperf_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(nperf_core PUBLIC
    CUDA::cudart
    NCCL::nccl
    NVML::nvml
)

# nlohmann_json is header-only, just add include path (avoids export issues)
target_include_directories(nperf_core PRIVATE
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

# Coordination library (separate for optional MPI)
add_library(nperf_coordination STATIC
    src/coordination/coordinator.cpp
    src/coordination/socket_coordinator.cpp
    src/coordination/nccl_bootstrap_coordinator.cpp
)

if(NPERF_BUILD_MPI)
    target_sources(nperf_coordination PRIVATE
        src/coordination/mpi_coordinator.cpp
    )
    target_link_libraries(nperf_coordination PUBLIC MPI::MPI_CXX)
    target_compile_definitions(nperf_coordination PUBLIC NPERF_HAS_MPI)
endif()

target_link_libraries(nperf_coordination PUBLIC nperf_core)

# Main library (combines all, including engine which needs coordination)
add_library(nperf STATIC
    src/lib/libnperf.cpp
    src/core/engine.cpp
)
target_link_libraries(nperf PUBLIC nperf_coordination)

# CLI library (for testing the parser)
add_library(nperf_cli_lib STATIC
    src/cli/parser.cpp
)
target_link_libraries(nperf_cli_lib PUBLIC
    nperf
    CLI11::CLI11
)

# CLI executable
add_executable(nperf_cli
    src/cli/main.cpp
)
target_link_libraries(nperf_cli PRIVATE
    nperf_cli_lib
)
set_target_properties(nperf_cli PROPERTIES OUTPUT_NAME nperf)

# Tests
if(NPERF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(NPERF_BUILD_EXAMPLES)
    # add_subdirectory(examples)
endif()

# Install targets with proper exports
install(TARGETS nperf nperf_core nperf_coordination
    EXPORT nperfTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# CLI is installed separately (not exported as library target)
install(TARGETS nperf_cli
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/nperf
    DESTINATION include
)

install(FILES ${CMAKE_BINARY_DIR}/include/nperf/version.h
    DESTINATION include/nperf
)

# Export targets for find_package support
install(EXPORT nperfTargets
    FILE nperfTargets.cmake
    NAMESPACE nperf::
    DESTINATION lib/cmake/nperf
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/nperfConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/nperfConfig.cmake"
    INSTALL_DESTINATION lib/cmake/nperf
)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/nperfConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/nperfConfig.cmake"
    "${CMAKE_BINARY_DIR}/nperfConfigVersion.cmake"
    DESTINATION lib/cmake/nperf
)

# Install CMake modules needed by config
install(FILES
    "${CMAKE_SOURCE_DIR}/cmake/FindNCCL.cmake"
    "${CMAKE_SOURCE_DIR}/cmake/FindNVML.cmake"
    DESTINATION lib/cmake/nperf
)
