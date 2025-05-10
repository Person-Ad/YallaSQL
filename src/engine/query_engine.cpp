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


}

void QueryEngine::executeDuckDB(std::string& query) {
    // Ensure the directory exists
    auto now = std::chrono::system_clock::now();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::string csvPath = YallaSQL::resultDir + "/duck_output_" + std::to_string(millis) + ".csv";
    // execute the query
    query.erase(query.find(';'), 1);
    auto queryRes = db_->duckdb().Query("COPY (" + query + ") TO '"+ csvPath  +"';");

    if (queryRes->HasError()) 
        throw QueryEngineError("Query execution failed: " + queryRes->GetError());

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


void QueryEngine::executeLogicalPlan(const std::string& query, std::string filename) {
    if(!db_) db_ = DB::getInstance();

    try {
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
        myExecutor.execute(*logicalPlan, planner, filename);

        db_->duckdb().Commit();


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

std::string QueryEngine::execute(std::string query, std::string filename) {
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
        
        if (lowerQuery.find("use") == 0) {
            useDB(query);
            return "Database switched successfully";
        }
        
        if(db_ == nullptr) 
            db_ = DB::getInstance();

        if (lowerQuery.find("duckdb") == 0) {
            DB::setPath(dbPath, true);
            query = query.substr(6); // length of "duckdb"
            executeDuckDB(query);
        }
        else if (lowerQuery.find("explain") == 0) {
            DB::setPath(dbPath, false);
            query = query.substr(7); // length of "explain"
            result = getLogicalPlan(query);
            saveQueryResult(result);
            
        }
        else {
            DB::setPath(dbPath, false);
            executeLogicalPlan(query, filename);
        }
        
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