# nperf Test Suite
# Google Test-based comprehensive test coverage

include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Coverage support (GCC/Clang only)
option(NPERF_COVERAGE "Enable code coverage" OFF)

if(NPERF_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Code coverage enabled")
        set(COVERAGE_FLAGS "--coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    else()
        message(WARNING "Code coverage only supported on GCC/Clang")
    endif()
endif()

# Common include directory for test utilities
set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# Unit Tests (No GPU required)
# ============================================================================
add_executable(nperf_unit_tests
    test_main.cpp
    unit/test_types.cpp
    unit/test_config.cpp
    unit/test_results.cpp
    unit/test_metrics.cpp
    unit/test_cli_parser.cpp
    unit/test_formatters.cpp
    unit/test_verifier_logic.cpp
)

target_include_directories(nperf_unit_tests PRIVATE
    ${TEST_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

target_link_libraries(nperf_unit_tests PRIVATE
    nperf_cli_lib
    GTest::gtest
)

# ============================================================================
# Integration Tests (GPU required)
# ============================================================================
add_executable(nperf_integration_tests
    test_main.cpp
    integration/test_cuda_timer.cpp
    integration/test_device_buffer.cpp
    integration/test_memory_manager.cpp
    integration/test_collective.cpp
    integration/test_graph.cpp
    integration/test_verifier.cpp
    integration/test_topology.cpp
    integration/test_coordinators.cpp
    integration/test_engine.cpp
)

target_include_directories(nperf_integration_tests PRIVATE
    ${TEST_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

target_link_libraries(nperf_integration_tests PRIVATE
    nperf
    GTest::gtest
    CUDA::cudart
)

# ============================================================================
# CTest Integration
# ============================================================================
include(GoogleTest)

# Unit tests can always run
gtest_discover_tests(nperf_unit_tests
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "unit"
)

# Integration tests require GPU - label them for conditional execution
gtest_discover_tests(nperf_integration_tests
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "integration;gpu"
)

# ============================================================================
# Coverage Report Target
# ============================================================================
if(NPERF_COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
    if(LCOV AND GENHTML)
        add_custom_target(coverage
            COMMAND ${LCOV} --capture --directory . --output-file coverage.info --ignore-errors mismatch
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/tests/*' '*/_deps/*' --output-file coverage.info --ignore-errors unused
            COMMAND ${GENHTML} coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report in coverage_report/"
            DEPENDS nperf_unit_tests nperf_integration_tests
        )
        message(STATUS "Coverage target available: make coverage")
    else()
        message(WARNING "lcov/genhtml not found - coverage target not available")
    endif()
endif()

# ============================================================================
# Convenience Targets
# ============================================================================

# Run only unit tests (no GPU required)
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --output-on-failure
    DEPENDS nperf_unit_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests"
)

# Run only integration tests (requires GPU)
add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --output-on-failure
    DEPENDS nperf_integration_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running integration tests (requires GPU)"
)
