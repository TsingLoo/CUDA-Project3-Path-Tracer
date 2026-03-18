cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
add_subdirectory(external/tinygltf)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/external/tinygltf/examples/common)  # tinyexr.h

# ============================================================================
# OptiX SDK (optional — hardware RT Core acceleration)
# ============================================================================
find_package(OptiX QUIET)
if(OptiX_FOUND)
    message(STATUS "OptiX found at: ${OptiX_INCLUDE_DIR}")
    include_directories(${OptiX_INCLUDE_DIR})
    set(OPTIX_ENABLED ON)

    # Compile OptiX device programs (.cu -> .ptx)
    set(OPTIX_PTX_INPUT "${CMAKE_SOURCE_DIR}/src/optix_programs.cu")
    set(OPTIX_PTX_OUTPUT "${CMAKE_BINARY_DIR}/bin/optix_programs.ptx")

    add_custom_command(
        OUTPUT ${OPTIX_PTX_OUTPUT}
        COMMAND ${CMAKE_CUDA_COMPILER}
            -ptx
            -std=c++17
            --use_fast_math
            -ccbin "${CMAKE_CXX_COMPILER}"
            "-I${OptiX_INCLUDE_DIR}"
            "-I${CMAKE_SOURCE_DIR}/src"
            "-I${CMAKE_SOURCE_DIR}/external/include"
            "-I${CMAKE_SOURCE_DIR}/external/tinygltf/examples/common" # tinyexr.h
            "-I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            -o ${OPTIX_PTX_OUTPUT}
            ${OPTIX_PTX_INPUT}
        DEPENDS ${OPTIX_PTX_INPUT}
                ${CMAKE_SOURCE_DIR}/src/optix_params.h
        COMMENT "Compiling OptiX programs to PTX..."
    )
    add_custom_target(optix_ptx ALL DEPENDS ${OPTIX_PTX_OUTPUT})
else()
    message(STATUS "OptiX SDK not found — building without RT Core acceleration")
    set(OPTIX_ENABLED OFF)
endif()

set(headers
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/utilities.h
    src/dispersion.h
    src/optix_params.h
    src/optix_renderer.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/tinyexr_impl.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/utilities.cpp
    src/optix_renderer.cu
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    tinygltf
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

# OptiX compile definitions
if(OPTIX_ENABLED)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_OPTIX=1)
    add_dependencies(${CMAKE_PROJECT_NAME} optix_ptx)
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_OPTIX=0)
endif()

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
