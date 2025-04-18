#ifndef YALLASQL_DB_TABLE
#define YALLASQL_DB_TABLE

#include "duckdb/main/connection.hpp"
#include "logger.hpp"
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cstdint>

enum class DataType : uint8_t {
    INT,
    FLOAT,
    DATETIME,
    STRING
};

struct Column {
    std::string name;
    DataType type : 2;    // 2 bits for DataType
    bool isPk : 1;        // 1 bit for primary key
    bool isFk : 1;        // 1 bit for foreign key
    uint8_t padding : 4;  // Explicit padding to align to byte boundary

    Column(std::string name_, DataType t, bool pk = false, bool fk = false)
        : name(std::move(name_)), type(t), isPk(pk), isFk(fk), padding(0) {}

    // Move constructor
    Column(Column&&) = default;
    Column& operator=(Column&&) = default;

    // Delete copy to prevent unintended copies
    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;
};

class Table {
    quill::Logger* logger_;
    duckdb::Connection& con;
    
public:
    std::string name;
    std::string path;
    std::unordered_map<std::string, Column> columns;
    std::unordered_map<std::string, const Column*> pkColumns; //
    std::unordered_map<const Table*, std::vector<std::pair<const Column*, Column*>>> fkColumns; // Table: [(reference_col, foreigen_col)]
    
    Table(std::string tableName, std::string filePath, duckdb::Connection& conn)
        : name(std::move(tableName)), 
          path(std::move(filePath)), 
          con(conn),
          logger_(YALLASQL::getLogger("")) {
        loadDuckDBTable();
        inferDuckDBSchema();
    }

    // Delete copy constructor and assignment
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;

    // Move constructor and assignment
    Table(Table&&) = default;
    Table& operator=(Table&&) = default;

    void setupForeignKeys(const std::unordered_map<std::string, Table>& tables) {
        for (const auto& [tableName, table] : tables) {
            if (tableName == name) continue;

            std::vector<std::pair<const Column*, Column*>> tableLinks;
            tableLinks.reserve(table.pkColumns.size());

            for (const auto& [pkName, pkCol] : table.pkColumns) {
                std::string cleanPkName = generateFkName(tableName, pkName);
                auto it = columns.find(cleanPkName);
                if (it != columns.end()) {
                    tableLinks.emplace_back(pkCol, &it->second);
                }
            }

            if (tableLinks.size() == table.pkColumns.size()) {
                fkColumns[&table] = std::move(tableLinks);
                LOG_INFO(logger_, "Detected Foreign Key Relationship where {} references {}", name, tableName);
            }
        }
    }

    void reCreateDuckDBTable() {
        // 1. drop if exist
        auto result = con.Query("DROP TABLE IF EXISTS \"" + name + "\"");

        // recreate table with all schema & 
        std::string query = "CREATE TABLE \"" + name + "\" (\n";
        // Columns
        size_t idx = 0;
        for (const auto& [colName, col] : columns) {
            query += "  \"" + colName + "\" " + dataTypeToSQL(col.type);
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
            query += ")\n";
        }

        query += ");";
        
        
        result = con.Query(query);

        if(result->HasError()) 
            LOG_ERROR(logger_, "Failed to recreate table {} query: {}: {}", name, query, result->GetError());
        else 
            LOG_INFO(logger_, "Success in Recreate table {}", name);
    }

private:


    void loadDuckDBTable() {
        con.Query("CREATE OR REPLACE TABLE " + name + 
                  " AS SELECT * FROM read_csv_auto('" + path + "')");
        LOG_INFO(logger_, "Loaded CSV {} as table {}", path, name);
    }

    void inferDuckDBSchema() {
        auto result = con.Query("PRAGMA table_info(" + name + ")");
        if (result->HasError()) {
            LOG_ERROR(logger_, "Failed to infer schema for table {}: {}", name, result->GetError());
            return;
        }

        columns.reserve(result->RowCount());
        for (size_t idx = 0; idx < result->RowCount(); ++idx) {
            std::string colName = result->GetValue(1, idx).ToString();
            std::string colType = result->GetValue(2, idx).ToString();
            
            bool isPk = colName.ends_with("(P)");
            
            auto [it, inserted] = columns.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(colName),
                std::forward_as_tuple(colName, inferDataType(colType), isPk)
            );

            if (isPk) {
                pkColumns.emplace(colName, &it->second);
            }
        }
    }

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

    [[nodiscard]] std::string generateFkName(std::string_view tableName, std::string_view pkName) const {
        std::string cleanPkName(pkName);
        if (cleanPkName.ends_with("(P)")) {
            cleanPkName = cleanPkName.substr(0, cleanPkName.length() - 3);
        }
        if (cleanPkName.ends_with(" ")) {
            cleanPkName.pop_back();
        }
        return std::string(tableName) + "_" + cleanPkName;
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
    
};

#endif