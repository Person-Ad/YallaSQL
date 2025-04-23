#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "engine/query_engine.hpp"
#include "engine/executor_engine.hpp"
#include "utils.hpp"

using namespace duckdb;
using namespace YallaSQL::UTILS;


void QueryEngine::useDB(const std::string& input) {
    // try {
        size_t space_pos = input.find(' ');

        std::string remainder = input.substr(space_pos + 1);
        size_t next_sep = remainder.find(' ');
               next_sep = next_sep == std::string::npos ? remainder.find(';') : next_sep;

        dbPath = remainder.substr(0, next_sep);


        // MEASURE_EXECUTION_TIME_LOGGER(logger_, "use db", DB::setPath(path));
        // DB::setPath(path);
        // if(db_ == nullptr) db_ = DB::getInstance();
    // } catch (const std::runtime_error& e) {
    //     LOG_ERROR(logger_, "Failed to switch database: {}", e.what());
    //     throw; // Re-throw to let the caller handle
    // }
}

QueryEngine::QueryResult QueryEngine::executeDuckDB(const std::string& query) {
    QueryResult result {true, ""};
    unique_ptr<MaterializedQueryResult> queryRes;
    
    queryRes = db_->duckdb().Query(query);


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
    if(!db_) db_ = DB::getInstance();

    try {
        QueryResult result{false, ""};
        db_->duckdb().BeginTransaction();

        // start timer
        Parser parser;
        parser.ParseQuery(query);
        auto statements = std::move(parser.statements);

        Planner planner(*db_->duckdb().context);
        planner.CreatePlan(std::move(statements[0]));


        Optimizer optimizer(*planner.binder, *db_->duckdb().context);
        unique_ptr<duckdb::LogicalOperator>  logicalPlan = std::move(optimizer.Optimize(std::move(planner.plan)));


        ExecutorEngine myExecutor;
        myExecutor.execute(*logicalPlan, planner);
        // logicalPlan = std::move(planner.plan);
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
        std::string lowerQuery = query;
        std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
        if (query.empty()) {
            throw QueryEngineError("Empty query provided");
        }

        // Route query
        QueryResult result;
        
        if (lowerQuery.find("use") != std::string::npos) {
            useDB(query);
            return "Database switched successfully";
        }

        if(db_ == nullptr) db_ = DB::getInstance();
        if (lowerQuery.find("duckdb") != std::string::npos) {
            DB::setPath(dbPath, true);
            result = executeDuckDB(query.substr(query.find("duckdb") + 6));
        }
        else {
            DB::setPath(dbPath, false);
            result = getLogicalPlan(query);
        }

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