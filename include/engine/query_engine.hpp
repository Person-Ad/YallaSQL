#ifndef YALLASQL_INCLUDE_ENGINE_HPP
#define YALLASQL_INCLUDE_ENGINE_HPP

#include <stdexcept>
#include <string>
#include "db/db.hpp"
#include "logger.hpp"

#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"

class QueryEngine {

private:
    DB* db_;
    quill::Logger* logger_;
    duckdb::Parser parser_;
    duckdb::Planner planner_;

    void useDB(const std::string& input);
    void cpuDB(const std::string& query);

    void getLogicalPlan(const std::string& query);

public:
    QueryEngine();

    std::string execute(std::string query);

};

#endif