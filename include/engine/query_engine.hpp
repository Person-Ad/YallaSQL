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
    std::string dbPath = "";

    void useDB(const std::string& input);
    void executeDuckDB(std::string& query);
    QueryResult getLogicalPlan(const std::string& query);
    void executeLogicalPlan(const std::string& query, std::string filename = "");
    void saveQueryResult(const QueryResult& result);


public:
    QueryEngine() :logger_(YallaSQL::getLogger("")) {}

    std::string execute(std::string query, std::string filename = "");

};

#endif