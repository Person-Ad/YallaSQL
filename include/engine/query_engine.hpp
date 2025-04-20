#ifndef YALLASQL_INCLUDE_ENGINE_HPP
#define YALLASQL_INCLUDE_ENGINE_HPP

#include <stdexcept>
#include <string>
#include "db/db.hpp"
#include "logger.hpp"

#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/executor.hpp"


class QueryEngineError : public std::runtime_error {
public:
    explicit QueryEngineError(const std::string& message) : std::runtime_error(message) {}
};

class QueryEngine {

private:
    struct QueryResult {
        bool isTabular;
        std::string content;
    };
    DB* db_ = nullptr;
    quill::Logger* logger_ = nullptr;
    const std::string resultsDir = "results";

    void useDB(const std::string& input);
    QueryResult executeDuckDB(const std::string& query);
    QueryResult getLogicalPlan(const std::string& query);
    void saveQueryResult(const QueryResult& result);


public:
    QueryEngine() :logger_(YallaSQL::getLogger("")) {}

    std::string execute(std::string query);

};

#endif