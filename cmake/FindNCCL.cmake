# FindNCCL.cmake
# Find NCCL (NVIDIA Collective Communications Library)
#
# This module defines:
#   NCCL_FOUND        - True if NCCL was found
#   NCCL_INCLUDE_DIRS - NCCL include directories
#   NCCL_LIBRARIES    - NCCL libraries
#   NCCL::nccl        - Imported target
#
# Hints:
#   NCCL_HOME         - Root directory of NCCL installation
#   NCCL_ROOT         - Alternative root directory

# Search paths
set(_NCCL_SEARCH_PATHS
    ${NCCL_HOME}
    $ENV{NCCL_HOME}
    ${NCCL_ROOT}
    $ENV{NCCL_ROOT}
    /usr/local/cuda
    /usr/local
    /usr
)

# Find include directory
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# Find library
find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${_NCCL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

# Get version from header
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _NCCL_VERSION_LINES
        REGEX "#define NCCL_(MAJOR|MINOR|PATCH)")

    foreach(_line ${_NCCL_VERSION_LINES})
        if(_line MATCHES "#define NCCL_MAJOR[ \t]+([0-9]+)")
            set(NCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "#define NCCL_MINOR[ \t]+([0-9]+)")
            set(NCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "#define NCCL_PATCH[ \t]+([0-9]+)")
            set(NCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    if(NCCL_VERSION_MAJOR AND NCCL_VERSION_MINOR AND NCCL_VERSION_PATCH)
        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
    VERSION_VAR NCCL_VERSION
)

if(NCCL_FOUND)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

    # Create imported target
    if(NOT TARGET NCCL::nccl)
        add_library(NCCL::nccl UNKNOWN IMPORTED)
        set_target_properties(NCCL::nccl PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "Found NCCL: ${NCCL_LIBRARY} (version ${NCCL_VERSION})")
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
