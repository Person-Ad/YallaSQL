#ifndef YALLASQL_DB_TABLE
#define YALLASQL_DB_TABLE

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cstdint>
// duckdb includes
#include "duckdb/main/connection.hpp"
// my includes
#include "logger.hpp"
#include "utils.hpp"
#include "enums/data_type_utils.hpp"
#include "db/column.hpp"

using namespace YallaSQL::UTILS;
class OurDuck;

class Table {
    friend class OurDuck;
    quill::Logger* logger_;
    duckdb::Connection& con;
public:
    std::string name;
    std::string path;
    size_t rowBytes = 0;    // total bytes per row
    uint32_t numCols = 0;   
    std::vector<DataType> columnsType;
    std::vector<std::shared_ptr<Column>> columnsOrdered; // used for perserving order in csv file   
    std::vector<std::string> csvNameColumn; // used for perserving order in csv file
    std::unordered_map<std::string, std::shared_ptr<Column>> columns;
    std::unordered_map<std::string, std::shared_ptr<Column>> pkColumns; //
    std::unordered_map<const Table*, std::vector<std::pair<std::shared_ptr<Column>, std::shared_ptr<Column>>>> fkColumns; // Table: [(reference_col, foreigen_col)]
    
    Table(std::string tableName, std::string filePath, duckdb::Connection& conn)
        : name(std::move(tableName)), 
          path(std::move(filePath)), 
          con(conn),
          logger_(YallaSQL::getLogger("")) {
        inferDBSchema();
    }

    // Delete copy constructor and assignment
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    // Move constructor and assignment
    Table(Table&&) = default;
    Table& operator=(Table&&) = default;
    // foreach other tables get the Forigen Key that reference this table
    void setupForeignKeys(const std::unordered_map<std::string, Table *>& tables);

    void reCreateDuckDBTable(bool insertInDuck = false) {
        // create table in duckdb 
        const std::string query = insertInDuck ? generateSimpleCreateTableQuery() : generateCreateTableQuery();
        auto result = con.Query(query);
        
        if(result->HasError()) 
            LOG_ERROR(logger_, "Failed to recreate table {} query: {}: {}", name, query, result->GetError());
        else 
            LOG_INFO(logger_, "Recreated table {}", name);

        // insert values into table
        if(insertInDuck) {
            result = con.Query("INSERT INTO \"" + name + "\" SELECT * FROM read_csv('" + path + "', header=True, sample_size=10)");

            if (result->HasError())
                LOG_ERROR(logger_, "Failed to load data into table {} from {} where schema {}: {}", name, query, path, result->GetError());
            else
                LOG_INFO(logger_, "loaded data into table {} from {}", name, path);
        }

    }

private:
    void inferDBSchema();

    [[nodiscard]] DataType inferDataType(std::string_view colType) const noexcept {
        if (colType.find("INT") != std::string_view::npos ||
            colType.find("BIGINT") != std::string_view::npos) {
            return DataType::INT;
        }
        if (colType.find("FLOAT") != std::string_view::npos ||
            colType.find("DOUBLE") != std::string_view::npos ||
            colType.find("REAL") != std::string_view::npos) {
            return DataType::FLOAT;
        }
        if (colType.find("DATE") != std::string_view::npos ||
            colType.find("TIMESTAMP") != std::string_view::npos) {
            return DataType::DATETIME;
        }
        return DataType::STRING;
    }

    [[nodiscard]] inline std::string generateFkName(std::string_view tableName, std::string_view pkName) const {
        // if (cleanPkName.ends_with("(P)")) {
        //     cleanPkName = cleanPkName.substr(0, cleanPkName.length() - 3);
        // }
        // if (cleanPkName.ends_with(" ")) {
        //     cleanPkName.pop_back();
        // }
        return std::string(tableName) + "_" + std::string(pkName);
    }


    [[nodiscard]] std::string dataTypeToSQL(DataType type) const noexcept {
        switch (type) {
            case DataType::INT: return "INTEGER";
            case DataType::FLOAT: return "FLOAT";
            case DataType::DATETIME: return "TIMESTAMP";
            case DataType::STRING: return "VARCHAR";
            default: return "VARCHAR";
        }
    }

    [[nodiscard]] std::string dataTypeToString(DataType type) const noexcept {
        switch (type) {
            case DataType::INT: return "INT";
            case DataType::FLOAT: return "FLOAT";
            case DataType::DATETIME: return "DATETIME";
            case DataType::STRING: return "STRING";
            default: return "STRING";
        }
    }

    [[nodiscard]] std::string generateCreateTableQuery() const {
        // recreate table with all schema &
        std::string query = "CREATE TABLE \"" + name + "\" (\n";
        // Columns
        size_t idx = 0;
        for (const auto& col : columnsOrdered) {
            query += "  \"" + col->name + "\" " + dataTypeToSQL(col->type);
            if (idx++ < columns.size() - 1) query += ",";
            query += "\n";
        }

        // Primary Key
        if (!pkColumns.empty()) {
            query += ",  PRIMARY KEY (";
            idx = 0;
            for (const auto& [colName, _] : pkColumns) {
                query += "\"" + colName + "\"";
                if (idx++ < pkColumns.size() - 1)  query += ", ";
            }
            query += ")\n";
        }

        // Foreign Keys
        idx = 0;
        for (const auto& [refTable, links] : fkColumns) {
            query += ",  FOREIGN KEY (";
            size_t colIdx = 0;
            for (const auto& [_, fkCol] : links) {
                query += "\"" + fkCol->name + "\"";
                if (colIdx++ < links.size() - 1) query += ", ";
            }

            query += ") REFERENCES \"" + refTable->name + "\" (";
            colIdx = 0;
            for (const auto& [pkCol, _] : links) {
                query += "\"" + pkCol->name + "\"";
                if (colIdx++ < links.size() - 1) query += ", ";
            }
            query += ") \n";
        }

        query += ");";

        return query;
    }


    [[nodiscard]] std::string generateSimpleCreateTableQuery() const {
        // recreate table with all schema &
        std::string query = "CREATE TABLE \"" + name + "\" (\n";
        // Columns
        size_t idx = 0;
        for (const auto& col : columnsOrdered) {
            query += "  \"" + col->name + "\" " + dataTypeToSQL(col->type);
            if (idx++ < columns.size() - 1) query += ",";
            query += "\n";
        }

        // Primary Key
        if (!pkColumns.empty()) {
            query += ",  PRIMARY KEY (";
            idx = 0;
            for (const auto& [colName, _] : pkColumns) {
                query += "\"" + colName + "\"";
                if (idx++ < pkColumns.size() - 1)  query += ", ";
            }
            query += ")\n";
        }

        // Foreign Keys
        query += ");";

        return query;
    }
    
};

#endif