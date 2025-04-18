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

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED CONFIG)
find_package(quill REQUIRED CONFIG)

# for duckdb
link_directories(vendor/duckdb/build/release/src)


# Define source directories
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Gather source files
set(CLI_SOURCES
    "${SOURCE_DIR}/cli/shell.cpp"
    "${SOURCE_DIR}/db/db.cpp"
    "${SOURCE_DIR}/engine/query_engine.cpp"
)

# Set include directories for the target
include_directories(    
    include
    vendor/duckdb/src/include
)

add_executable(yallasql_cli
    "${SOURCE_DIR}/cli/main.cpp"
    ${CLI_SOURCES}
)


# Link libraries
target_link_libraries(yallasql_cli
    PRIVATE
        CUDA::cudart
        replxx::replxx
        quill::quill
        duckdb
)
