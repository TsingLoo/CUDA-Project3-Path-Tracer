# FindOptiX.cmake
# Finds the NVIDIA OptiX SDK (header-only, 7.0+)
# Sets: OptiX_FOUND, OptiX_INCLUDE_DIR

# Check environment variable first
if(DEFINED ENV{OptiX_INSTALL_DIR})
    set(_optix_search_dirs "$ENV{OptiX_INSTALL_DIR}")
endif()

# Standard install paths
list(APPEND _optix_search_dirs
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
    "/usr/local/NVIDIA-OptiX-SDK-9.1.0"
    "/usr/local/NVIDIA-OptiX-SDK-8.0.0"
    "/usr/local/NVIDIA-OptiX-SDK-7.7.0"
)

find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    PATHS ${_optix_search_dirs}
    PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR
)

if(OptiX_FOUND)
    message(STATUS "Found OptiX: ${OptiX_INCLUDE_DIR}")
endif()

mark_as_advanced(OptiX_INCLUDE_DIR)
