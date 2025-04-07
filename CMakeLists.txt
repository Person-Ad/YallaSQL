cmake_minimum_required(VERSION 3.18)
project(yallasql LANGUAGES CXX CUDA)

# --- Linenoise Setup ---
include_directories(    
    include/cli
)

set(STATES_SOURCES
        include/cli/color.hpp
        include/cli/command.hpp
        include/cli/cursor.hpp
        include/cli/shell.hpp
        src/cli/shell.cpp
)


# --- packages Support ---
find_package(CUDAToolkit REQUIRED)
find_package(replxx REQUIRED)
# --- Main Executable ---

add_executable(yallasql_cli
    src/cli/main.cpp

    ${STATES_SOURCES}
)

target_link_libraries(yallasql_cli PRIVATE CUDA::cudart replxx)  # Uncomment later