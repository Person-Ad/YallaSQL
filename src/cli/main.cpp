#include <iostream>
#include <cuda_runtime.h>

#include "cli/shell.hpp"
#include "utils/macros.hpp"

#define YALLASQL_DEBUG true

int main() {
    CUDA_CHECK(cudaFree(0));
    YallaSQLShell app;
    app.run();
    return 0;
}
