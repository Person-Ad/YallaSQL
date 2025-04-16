#include "engine/query_engine.hpp"
#include "utils.hpp"

using namespace YALLASQL::UTILS;

QueryEngine::QueryEngine() {
    db_ = DB::getInstance();
    logger_ = YALLASQL::getLogger("./logs/engine");
}

void QueryEngine::useDB(const std::string input) {
    std::string path;
    size_t space_pos = input.find(' ');
    
    if (space_pos != std::string::npos)  
        path = input.substr(space_pos + 1);
    else
        throw std::runtime_error("No Direcroty was written");

    MEASURE_EXECUTION_TIME_LOGGER(logger_, "use db", DB::setPath(path));
}

std::string QueryEngine::execute(std::string query) {
    if(query.find("USE") == 0) {
        useDB(query);
    }
    return "success";
}