#include <filesystem>
#include <fstream>
#include <stdexcept>
namespace fs = std::filesystem;

#include "db.hpp"

DB* DB::db_ = nullptr;
std::mutex DB::mutex_;

DB *DB::getInstance(const std::string& path) {
    // 1. check if path exists and it's directory
    if (!fs::exists(path)) {
        throw std::runtime_error("Database folder does not exist: " + path);
    }
    if (!fs::is_directory(path)) {
        throw std::runtime_error("Database path is not a directory: " + path);
    }
    
    // 2. lock database instance
    std::lock_guard<std::mutex> lock(mutex_);
    // 3. create new database instance if none exists || change path
    if(db_ == nullptr) {
        db_ = new DB(path);
    } else {
        db_->path_ = path;
        db_->refreshTables();
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