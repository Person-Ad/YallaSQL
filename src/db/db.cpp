#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "logger.hpp"
#include "db/db.hpp"

namespace fs = std::filesystem;

DB* DB::db_ = nullptr;
std::mutex DB::mutex_;


void DB::setPath(const std::string& path) {
    // lock database instance
    std::lock_guard<std::mutex> lock(mutex_);
    // create directory if it doesn't exist
    if (!fs::exists(path)) {
        throw std::runtime_error("Failed to find database directory: " + path);
    }
    // verify it's a directory (in case path exists but is a file)
    else if (!fs::is_directory(path)) 
        throw std::runtime_error("Database path exists but is not a directory: " + path);

    // create new database instance if none exists || change path
    if(db_ == nullptr) {
        db_ = new DB(path);
    } else {
        db_->path_ = path;
        db_->refreshTables();
    }
}

DB *DB::getInstance() {
    if (db_ == nullptr) {
        setPath();
    }
    return db_;
}

void DB::reCreateLinkedDuckDB(const std::string& tableName, std::unordered_map<std::string, bool>& created) {
    if (created.find(tableName) != created.end() && created[tableName]) 
        return;
    

    // ensure referenced tables are created first
    for (const auto& [refTable, _] : tables_.find(tableName)->second->fkColumns) {
        if (created.find(refTable->name) == created.end() || !created[refTable->name]) {
            reCreateLinkedDuckDB(refTable->name, created);
        }
    }

    // Recreate the table
    MEASURE_EXECUTION_TIME_LOGGER(logger_, "reCreateDuckDBTable", tables_.find(tableName)->second->reCreateDuckDBTable());
    created[tableName] = true;
}

void DB::refreshTables() {
    tables_.clear();
    
    try {
        for (const auto& entry : fs::directory_iterator(path_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                std::string tableName = entry.path().stem().string();
                std::string filePath = entry.path().string();
                // 1. read all tables --> DuckDB read csv 
                // 2. infer Column Names & Datatypes & PKs from DuckDB (i.e Basic Schema)
                tables_.emplace(tableName, new Table(tableName, filePath, duckdb()));
            }
        }
        
        // 3. infer FKs from all tables & their primary Key
        for(auto [tableName, table]: tables_) 
            MEASURE_EXECUTION_TIME_LOGGER(logger_, "SetupFoginKey", table->setupForeignKeys(tables_));
        // 4. Recreate DuckDB tables with all constraints & schema infered using ToplogicalSort
        std::unordered_map<std::string, bool> created;
        for(auto& [tableName, table]: tables_) 
            MEASURE_EXECUTION_TIME_LOGGER(logger_, "reCreateLinkedDuckDB", reCreateLinkedDuckDB(tableName, created));

    } 
    catch (std::exception& e) {
        LOG_ERROR(logger_, "Failed to load: {}", e.what());
        std::cout << "Failed to load: " << e.what() << "\n";
    }
}


DB::~DB () {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_ != nullptr) {
        delete db_;
        db_ = nullptr;
    }
}