#ifndef YALLASQL_INCLUDE_DB_HPP
#define YALLASQL_INCLUDE_DB_HPP

#include<string>
#include <mutex>
#include<vector>
#include <ranges>
#include "logger.hpp"
#include<map>

const std::string DEFAULT_DATASET_PATH = "./dataset";

class DB {

private:

    static DB* db_;
    std::string path_;
    quill::Logger* logger_;
    static std::mutex mutex_;
    std::map<std::string, std::string> tablesPaths_;

    // private singelton constructor
    DB(const std::string path): path_(path) { 
        refreshTables();
        logger_ = YALLASQL::getLogger("./logs/database");
    }
    // reload tablesPaths_
    void refreshTables();

public:
    // db can't be cloned
    DB(const DB &) = delete;
    // db can't be assigned
    void operator=(const DB &) = delete;
    // set database path
    static void setPath(const std::string& path = DEFAULT_DATASET_PATH);
    // control access to db instance
    static DB *getInstance();
    // get path of database
    std::string path() const { return path_; }

    ~DB();
};

#endif // YALLASQL_INCLUDE_DB_HPP