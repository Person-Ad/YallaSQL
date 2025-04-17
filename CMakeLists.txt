cmake_minimum_required(VERSION 3.18)
project(yallasql LANGUAGES CXX CUDA)


link_directories(vendor/duckdb/build/release/src)

include_directories(    
    include
    vendor/duckdb/src/include
)
    
set(CLI_SOURCES
    src/cli/shell.cpp
    src/db/db.cpp
    src/engine/query_engine.cpp
)
    
# --- packages Support ---
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED)
find_package(quill REQUIRED)
# --- Main Executable ---

add_executable(yallasql_cli
    src/cli/main.cpp
    ${CLI_SOURCES}
)

target_link_libraries(yallasql_cli PRIVATE CUDA::cudart replxx quill::quill duckdb)  