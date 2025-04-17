#include "engine/query_engine.hpp"
#include "utils.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/planner/binder.hpp"
using namespace duckdb;
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
    } else if (query.find("SHOW") == 0) {

    } else if (query.find("SELECT") == 0) {
        db_->duckdb().BeginTransaction();
        // 1. Parse SQL → AST
        Parser parser;
        parser.ParseQuery(query);
        auto statements = std::move(parser.statements);
        std::cout << "Parsed AST: " << statements[0]->ToString() << "\n";

        // 2. AST -> logical plan
        Planner planner(*db_->duckdb().context);
        planner.CreatePlan(std::move(statements[0]));
        auto logical_plan = std::move(planner.plan);

        // Print Logical plan
        std::cout << "Logical Plan:\n" << logical_plan->ToString() << "\n";

        db_->duckdb().Commit();
    }
    return "success";
}