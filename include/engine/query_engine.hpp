#ifndef YALLASQL_INCLUDE_ENGINE_HPP
#define YALLASQL_INCLUDE_ENGINE_HPP

#include <stdexcept>
#include <string>
#include "db/db.hpp"
#include "logger.hpp"

class QueryEngine {

private:
    DB* db_;
    quill::Logger* logger_;

    void useDB(const std::string input);

public:
    QueryEngine();

    std::string execute(std::string query);

};

#endif