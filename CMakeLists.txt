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
# use seperate compiler for cuda files only
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # Ensure lineinfo for all builds

 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -g")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED CONFIG)
find_package(quill REQUIRED CONFIG)

# for duckdb
 link_directories(vendor/duckdb/build/debug/src)
# link_directories(vendor/duckdb/build/release/src)


# Define source directories
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(ALL_SOURCES
    "${SOURCE_DIR}/db/db.cpp"
    "${SOURCE_DIR}/db/table.cpp"
    # files
    "${SOURCE_DIR}/batch.cu"
    "${SOURCE_DIR}/kernels/binary_operators_kernel.cu"
    "${SOURCE_DIR}/kernels/cast_operators_kernel.cu"
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

add_executable(yallasql
    "main.cpp"
    ${ALL_SOURCES}
)
#add_compile_definitions(ASAN_OPTIONS="protect_shadow_gap=0:replace_intrin=0:detect_leaks=0")

# Link libraries
target_link_libraries(yallasql_cli
    PRIVATE
        asan 
        CUDA::cudart
        replxx::replxx
        quill::quill
        duckdb
)


# target_link_libraries(yallasql
#     PRIVATE
#         quill::quill
#         CUDA::cudart
#         duckdb
# )