cmake_minimum_required(VERSION 3.18)
project(yallasql LANGUAGES CXX CUDA)

find_package(quill REQUIRED)

include_directories(    
    include
)

set(CLI_SOURCES
        src/cli/shell.cpp
        src/db/db.cpp
)

# --- packages Support ---
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED)
# --- Main Executable ---

add_executable(yallasql_cli
    src/cli/main.cpp
    ${CLI_SOURCES}
)

target_link_libraries(yallasql_cli PRIVATE CUDA::cudart replxx quill::quill)  