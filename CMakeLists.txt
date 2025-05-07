cmake_minimum_required(VERSION 3.24)
project(yallasql
    LANGUAGES CXX CUDA
    VERSION 1.0.0
    DESCRIPTION "YallaSQL Database Project"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type
set(CMAKE_CUDA_ARCHITECTURES 86) 
# Configure output directories based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")
elseif(CMAKE_BUILD_TYPE STREQUAL "CUDADebug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cudadebug")
endif()

# Configure build-type specific flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "CUDADebug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED CONFIG)
find_package(quill REQUIRED CONFIG)

# Configure duckdb link directories based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    link_directories(vendor/duckdb/build/debug/src)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "CUDADebug")
    link_directories(vendor/duckdb/build/release/src)
endif()

# Define source directories
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(ALL_SOURCES
    "${SOURCE_DIR}/db/db.cpp"
    "${SOURCE_DIR}/db/table.cpp"
    # files
    "${SOURCE_DIR}/batch.cu"
    "${SOURCE_DIR}/kernels/binary_operators_kernel.cu"
    "${SOURCE_DIR}/kernels/comparison_operators_kernel.cu"
    "${SOURCE_DIR}/kernels/cast_operators_kernel.cu"
    "${SOURCE_DIR}/kernels/prefix_sum_kernel.cu"
    "${SOURCE_DIR}/kernels/move_rows_kernel.cu"
    "${SOURCE_DIR}/kernels/reduction_kernel.cu"
    "${SOURCE_DIR}/kernels/sum_douple_kernel.cu"
    "${SOURCE_DIR}/kernels/radix_sort_kernel.cu"
    "${SOURCE_DIR}/kernels/radix_sort_utils.cu"
    "${SOURCE_DIR}/kernels/radix_sort_uint32_kernel.cu"
    # main engine 
    "${SOURCE_DIR}/engine/query_engine.cpp"
    "${SOURCE_DIR}/engine/executor_engine.cpp"
    # Operators
    "${SOURCE_DIR}/engine/operators/operator.cpp"
    "${SOURCE_DIR}/engine/operators/get_operator.cpp"
    "${SOURCE_DIR}/engine/operators/expressions/expression.cpp"
)
# Gather source files
set(CLI_SOURCES
    "${SOURCE_DIR}/cli/shell.cpp"
)

# Set include directories for the target
include_directories(    
    include
    vendor
    vendor/duckdb/src/include
)

add_executable(yallasql_cli
    "${SOURCE_DIR}/cli/main.cpp"
    ${CLI_SOURCES}
    ${ALL_SOURCES}
)

# Link libraries
target_link_libraries(yallasql_cli
    PUBLIC
        CUDA::cudart
        replxx::replxx
        quill::quill
        duckdb
)

# Custom target for building Debug configuration
# Run `ninja debug` to build with CMAKE_BUILD_TYPE=Debug
# Custom targets
add_custom_target(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build . --config Debug
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building Debug configuration"
)

add_custom_target(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build . --config Release
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building Release configuration"
)

add_custom_target(cudadebug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=CUDADebug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build . --config CUDADebug
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building CUDADebug configuration"
)