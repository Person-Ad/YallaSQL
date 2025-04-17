#ifndef YALLASQL_DB_TABLE
#define YALLASQL_DB_TABLE

#include "duckdb/main/connection.hpp"
#include "logger.hpp"
#include <iostream>
#include <string>
#include <vector>

enum class DataType : u_int8_t {
    INT,
    FLOAT,
    DATETIME,
    STRING
};

struct Column {
    std::string name;
    DataType type : 2; // 2 bits for DataType (up to 4 values)
    u_int8_t isPk : 1;  // 1 bit for primary key flag
    u_int8_t isFk : 1;  // 1 bit for foreign key flag

    Column(std::string n, DataType t, bool pk = false, bool fk = false)
        : name(std::move(n)), type(t), isPk(pk), isFk(fk) {}
};


class Table {

public:
    std::string name;
    std::string path;
    std::vector<Column> columns;
    Column *pkColumn = nullptr;
    quill::Logger* logger_;
    duckdb::Connection& con;

    Table(const std::string tableName, const std::string filePath, duckdb::Connection& con): name(tableName), path(filePath), con(con) {
        logger_ = YALLASQL::getLogger("");
        
        loadDuckDBTable();
        inferDuckDBSchema();
        setPrimaryKeysDuckDB();
    }

private:
    void loadDuckDBTable() {
        con.Query("CREATE OR REPLACE TABLE " + name + 
                          " AS SELECT * FROM read_csv_auto('" + path + "')");
        
        LOG_INFO(logger_, "Loaded CSV {} as table {}", path, name);
        std::cout << "Loaded CSV " << path << " as table " << name << "\n";
    }

    void inferDuckDBSchema() {
        // Get the schema of the table
        auto result = con.Query("PRAGMA table_info(" + name + ")");
        if (result->HasError()) {
            LOG_ERROR(logger_, "Failed to infer schema for table {} because {}", name, result->GetError());
            return;
        }
        // Parse schema
        for (size_t idx = 0; idx < result->RowCount(); ++idx) {
            std::string colName = result->GetValue(1, idx).ToString();

            std::string colType = result->GetValue(2, idx).ToString();
            DataType dtype = inferDataType(colType);
            
            bool isPk = (colName.length() >= 3 && colName.substr(colName.length() - 3) == "(P)");
            columns.emplace_back(colName, dtype, isPk, false);
            // set reference to it
            if(isPk) 
                pkColumn = &columns.back();
        }    
    }
    // setup primary keys by updating constraints
    void setPrimaryKeysDuckDB() {
        if(pkColumn == nullptr) return; 

        std::string query = "ALTER TABLE " + name + \
                            " ADD CONSTRAINT \"pk_" + name + "\""\
                            " PRIMARY KEY (\"" + pkColumn->name + "\");";

        auto result = con.Query(query);
        
        if (result->HasError()) {
            LOG_ERROR(logger_, "Failed to set primary key for table {}: {}", name, result->GetError());
            std::cout << "Error setting primary key for table " << name << ": " << result->GetError() << "\n";
        } else {
            LOG_INFO(logger_, "Successfully set primary key for table {} on column: {}", name, pkColumn->name);
            std::cout << "Successfully set primary key for table " << name << " on columns: " << pkColumn->name << "\n";
        }

    }

    // input colType of duckdb output DataType
    DataType inferDataType(const std::string& colType) {
        if (colType.find("INT") != std::string::npos ||
            colType.find("BIGINT") != std::string::npos) 
            return DataType::INT;

        else if (colType.find("FLOAT") != std::string::npos ||
                colType.find("DOUBLE") != std::string::npos ||
                colType.find("REAL") != std::string::npos) 
            return DataType::FLOAT;
            
        else if (colType.find("DATE") != std::string::npos ||
                colType.find("TIMESTAMP") != std::string::npos) 
            return DataType::DATETIME;

        return DataType::STRING;
    }
}; 
#endif