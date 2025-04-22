#ifndef YALLASQL_INCLUDE_DB_HPP
#define YALLASQL_INCLUDE_DB_HPP

#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>
#include <ranges>
#include <unordered_map>
#include "logger.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
using namespace duckdb;

#include "db/table.hpp"

const std::string DEFAULT_DATASET_PATH = "default_dataset";

class DB {

private:

    static DB* db_;
    std::string path_;
    quill::Logger* logger_;
    static std::mutex mutex_;
    std::unordered_map<std::string, Table*> tables_;

    std::unique_ptr<DuckDB> duckdb_; // In-memory DB
    std::unique_ptr<Connection> con_;


    // private singelton constructor
    DB(const std::string path): path_(path) {
        duckdb_ = std::make_unique<DuckDB>(nullptr);
        con_ = std::make_unique<Connection>(*duckdb_);
        // to not optimize too much
        con_->Query(" SET disabled_optimizers = 'filter_pushdown,statistics_propagation';");
        con_->context->EnableProfiling();
        logger_ = YallaSQL::getLogger("./logs/database");
        refreshTables();
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
    // do topoligical sort to recreate DuckDB
    void reCreateLinkedDuckDB(const std::string &tableName, std::unordered_map<std::string, bool>& created);
    // get path of database
    std::string path() const { return path_; }
    // get connection to duckdb
    Connection& duckdb() const { return *con_; }
    // get table in db
    const Table* getTable(const std::string& tableName) const { 
        if(tables_.find(tableName) == tables_.end())
            throw std::runtime_error("YallaSQL can't find table: " + tableName);

        return tables_.find(tableName)->second; 
    }

    ~DB();
};

#endif // YALLASQL_INCLUDE_DB_HPP