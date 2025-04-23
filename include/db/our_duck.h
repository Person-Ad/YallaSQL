#pragma once


#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "logger.hpp"
using namespace duckdb;

class OurDuck {
public:
    std::unique_ptr<DuckDB> duckdb_;
    std::unique_ptr<Connection> connection;
    quill::Logger* logger_ = YallaSQL::getLogger("");
    OurDuck() {
        duckdb_ = std::make_unique<DuckDB>(nullptr);
        connection = std::make_unique<Connection>(*duckdb_);
        // to not optimize too much
        connection->Query(" SET disabled_optimizers = 'filter_pushdown,statistics_propagation';");
    }

    Connection& getConnection() const { return *connection; }
    ClientContext& getContext() const { return connection->context; }

    duckdb::Value getCsvSchema(const std::string& path) {
        auto result = connection->Query("SELECT COLUMNS FROM sniff_csv('" + path + "', sample_size=10)");
        if(result->HasError())  {
            LOG_ERROR(logger_, "Can't sniff csv from path {}: {}", path, result->GetError());
            throw std::runtime_error("Can't sniff csv from path: " + result->GetError());
        }

        auto schema = result->GetValue(0, 0); // it's only one row & col
        return schema;
    }
};