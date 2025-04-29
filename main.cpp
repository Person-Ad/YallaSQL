#include <iostream>
#include "engine/query_engine.hpp"

int main(int, char**){
    QueryEngine engine;
    // test cases
    engine.execute("use large_dataset");
    engine.execute("SELECT I, v, (I/2) as half, (I%2) as rem, 2.5*I+v+3 from (SELECT id as I, views as v from table_1);");
    
    std::cout << "Hello, from yallasql!\n";
}
