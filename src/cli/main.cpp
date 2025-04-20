#include <iostream>
#include "cli/shell.hpp"

#define YALLASQL_DEBUG true

int main() {
    YallaSQLShell app;
    app.run();
    return 0;
}
