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
        if (!fs::create_directories(path)) {
            throw std::runtime_error("Failed to create database directory: " + path);
        }
        // LOG_INFO(logger, "Created new database directory: {}", path);
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
        // LOG_WARNING("USE DEFAULT DATABASE in ./dataset folder");
        setPath();
    }
    return db_;
}

void DB::refreshTables() {
    tablesPaths_.clear();

    // iterate through all files in the directory
    for (const auto& entry : fs::directory_iterator(path_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string tableName = entry.path().stem().string();
            db_->tablesPaths_[tableName] = entry.path().string();
        }
    }
}


DB::~DB () {
    std::lock_guard<std::mutex> lock(mutex_);
    if (db_ != nullptr) {
        delete db_;
        db_ = nullptr;
    }
}