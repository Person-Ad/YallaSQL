#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "engine/query_engine.hpp"
#include "utils.hpp"

using namespace duckdb;
using namespace YALLASQL::UTILS;


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

QueryEngine::QueryResult QueryEngine::executeDuckDB(const std::string& query) {
    QueryResult result {true, ""};
    unique_ptr<MaterializedQueryResult> queryRes;
    
    MEASURE_EXECUTION_TIME_LOGGER(logger_, "cpu duckdb", 
        queryRes = db_->duckdb().Query(query);
    );


    if (queryRes->HasError()) 
        throw QueryEngineError("Query execution failed: " + queryRes->GetError());

    // if not tabular
    if(queryRes->RowCount() <= 1) 
        return {false, queryRes->ToString()};
    
    // save query result in csv format
    size_t nrows = queryRes->RowCount();
    size_t ncols = queryRes->ColumnCount();
    auto colnames = queryRes->names;
    // save header
    for (size_t colidx = 0; colidx < ncols; ++colidx)  {
        result.content += colnames[colidx];
        if(colidx < ncols - 1) result.content += ",";
    }
    result.content += "\n";
    // save values
    for (size_t idx = 0; idx < nrows; ++idx) {
        for (size_t colidx = 0; colidx < ncols; ++colidx) {
            result.content += queryRes->GetValue(colidx, idx).ToString();
            if(colidx < ncols - 1) result.content += ",";
        }
        result.content += "\n";
    }
    return result;
}

QueryEngine::QueryResult QueryEngine::getLogicalPlan(const std::string& query) {
    QueryResult result{false, ""};

    unique_ptr<duckdb::LogicalOperator> logicalPlan;
    try {
        Parser parser;
        Planner planner(*db_->duckdb().context);

        db_->duckdb().BeginTransaction();
        // start timer
        MEASURE_EXECUTION_TIME_LOGGER(logger_, "generating logical plan", 
            parser.ParseQuery(query);
            auto statements = std::move(parser.statements);
            
            planner.CreatePlan(std::move(statements[0]));
            logicalPlan = std::move(planner.plan);
        );
        // save content
        result.content = logicalPlan->ToString();

        db_->duckdb().Commit();

        return result;

    } catch (const duckdb::Exception& e) {
        db_->duckdb().Rollback();
        throw QueryEngineError("Logical plan generation failed: " + std::string(e.what()));
    } catch (const std::runtime_error& e) {
        db_->duckdb().Rollback();
        throw QueryEngineError("Logical plan generation failed: " + std::string(e.what()));
    }
}


void QueryEngine::saveQueryResult(const QueryResult& result) {
    // Ensure the directory exists
    std::filesystem::create_directories(resultsDir);

    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&timeNow);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = oss.str();

    // Choose extension
    std::string extension = result.isTabular ? ".csv" : ".txt";
    std::string filePath = resultsDir + "/result_" + timestamp + extension;

    // Write content
    std::ofstream out(filePath);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    out << result.content;
    out.close();
}

std::string QueryEngine::execute(std::string query) {
    try {
        // Trim query
        query.erase(0, query.find_first_not_of(" \t\n\r"));
        query.erase(query.find_last_not_of(" \t\n\r") + 1);
        if (query.empty()) {
            throw QueryEngineError("Empty query provided");
        }

        // Route query
        QueryResult result;
        
        if (query.find("USE") != std::string::npos) {
            useDB(query);
            return "Database switched successfully";
        }

        if (query.find("duckdb") != std::string::npos) 
            result = executeDuckDB(query.substr(query.find("duckdb") + 6));
        else 
            result = getLogicalPlan(query);
        

        saveQueryResult(result);
        
        return "Query executed successfully";
        
    } catch (const QueryEngineError& e) {
        LOG_ERROR(logger_, "Query execution failed: {}", e.what());
        return "Error: " + std::string(e.what());
    } catch (const std::exception& e) {
        LOG_ERROR(logger_, "Unexpected error: {}", e.what());
        return "Unexpected error: " + std::string(e.what());
    } catch (...) {
        LOG_ERROR(logger_, "Unknown error during query execution");
        return "Unknown error occurred";
    }
}