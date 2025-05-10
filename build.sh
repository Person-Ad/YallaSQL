cd vendor/duckdb && make release
mkdir build && cd buuild && cmake ..  && make release
./build/release/yallasql_cli