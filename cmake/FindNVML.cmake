# FindNVML.cmake
# Find NVML (NVIDIA Management Library)
#
# This module defines:
#   NVML_FOUND        - True if NVML was found
#   NVML_INCLUDE_DIRS - NVML include directories
#   NVML_LIBRARIES    - NVML libraries
#   NVML::nvml        - Imported target
#
# NVML is part of the CUDA toolkit and the driver package

# Search paths
set(_NVML_SEARCH_PATHS
    ${CUDAToolkit_ROOT}
    ${CUDA_TOOLKIT_ROOT_DIR}
    $ENV{CUDA_HOME}
    /usr/local/cuda
    /usr
)

# Find include directory
find_path(NVML_INCLUDE_DIR
    NAMES nvml.h
    HINTS ${_NVML_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
# Note: On Linux, libnvidia-ml.so is typically in /usr/lib or driver paths
# The stubs version is in CUDA toolkit
find_library(NVML_LIBRARY
    NAMES nvidia-ml
    HINTS ${_NVML_SEARCH_PATHS}
    PATH_SUFFIXES
        lib64
        lib64/stubs
        lib
        lib/stubs
        lib/x86_64-linux-gnu
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML
    REQUIRED_VARS NVML_LIBRARY NVML_INCLUDE_DIR
)

if(NVML_FOUND)
    set(NVML_LIBRARIES ${NVML_LIBRARY})
    set(NVML_INCLUDE_DIRS ${NVML_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET NVML::nvml)
        add_library(NVML::nvml UNKNOWN IMPORTED)
        set_target_properties(NVML::nvml PROPERTIES
            IMPORTED_LOCATION "${NVML_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NVML_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found NVML: ${NVML_LIBRARY}")
endif()

mark_as_advanced(NVML_INCLUDE_DIR NVML_LIBRARY)
