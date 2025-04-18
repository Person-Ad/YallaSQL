#include <stdexcept>

#include "engine/query_engine.hpp"
#include "utils.hpp"

using namespace duckdb;
using namespace YALLASQL::UTILS;

QueryEngine::QueryEngine() : db_(DB::getInstance()), planner_(*db_->duckdb().context) {
    logger_ = YALLASQL::getLogger("");
}

void QueryEngine::useDB(const std::string& input) {
    try {
        std::string path;
        size_t space_pos = input.find(' ');

        if (space_pos == std::string::npos || space_pos + 1 >= input.length()) {
            throw std::runtime_error("Invalid USE command: No directory specified");
        }

        path = input.substr(space_pos + 1);
        if (path.empty()) {
            throw std::runtime_error("Invalid USE command: Directory path is empty");
        }

        MEASURE_EXECUTION_TIME_LOGGER(logger_, "use db", DB::setPath(path));
    } catch (const std::runtime_error& e) {
        LOG_ERROR(logger_, "Failed to switch database: {}", e.what());
        throw; // Re-throw to let the caller handle
    }
}

void QueryEngine::cpuDB(const std::string& query) {
    try {
        unique_ptr<MaterializedQueryResult> result;
        MEASURE_EXECUTION_TIME_LOGGER(logger_, "cpu duckdb", 
            result = db_->duckdb().Query(query);
        );

        if (result->HasError()) 
            throw std::runtime_error("Query execution failed: " + result->GetError());
        std::cout << result->ToString() << '\n';

    } catch (const duckdb::Exception& e) {
        LOG_ERROR(logger_, "DuckDB query error: {}", e.what());
        throw std::runtime_error("DuckDB query error: " + std::string(e.what()));
    } catch (const std::runtime_error& e) {
        LOG_ERROR(logger_, "Query execution error: {}", e.what());
        throw; // Re-throw to let the caller handle
    }
}

void QueryEngine::getLogicalPlan(const std::string& query) {
    try {
        db_->duckdb().BeginTransaction();

        MEASURE_EXECUTION_TIME_LOGGER(logger_, "generating logical plan", 
            parser_.ParseQuery(query);
            auto statements = std::move(parser_.statements);
            
            planner_.CreatePlan(std::move(statements[0]));
            auto logical_plan = std::move(planner_.plan);

            std::cout << "Logical Plan:\n" << logical_plan->ToString() << "\n";
        );

        db_->duckdb().Commit();

    } catch (const duckdb::Exception& e) {
        db_->duckdb().Rollback();
        LOG_ERROR(logger_, "Logical plan generation failed: {}", e.what());
        throw std::runtime_error("Logical plan generation failed: " + std::string(e.what()));
    } catch (const std::runtime_error& e) {
        db_->duckdb().Rollback();
        LOG_ERROR(logger_, "Logical plan error: {}", e.what());
        throw; // Re-throw to let the caller handle
    }
}

std::string QueryEngine::execute(std::string query) {
    try {
        // Trim and validate query
        query.erase(0, query.find_first_not_of(" \t\n\r"));
        query.erase(query.find_last_not_of(" \t\n\r") + 1);
        if (query.empty()) {
            throw std::runtime_error("Empty query provided");
        }

        // Case-insensitive command checks
        std::string query_upper = query;
        std::transform(query_upper.begin(), query_upper.end(), query_upper.begin(), ::toupper);

        if (query_upper.find("DUCKDB") != std::string::npos) 
            cpuDB(query.substr(query_upper.find("DUCKDB") + 6));
        else if (query_upper.find("USE") != std::string::npos) 
            useDB(query);
        else 
            getLogicalPlan(query);
        
        return "Query executed successfully";
        
    } catch (const std::runtime_error& e) {
        LOG_ERROR(logger_, "Query execution failed: {}", e.what());
        return "Error: " + std::string(e.what());
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Unexpected error during query execution: {}", e.what());
        return "Unexpected error: " + std::string(e.what());
    } catch (...) {
        LOG_ERROR(logger_, "Unknown error during query execution");
        return "Unknown error occurred";
    }
}