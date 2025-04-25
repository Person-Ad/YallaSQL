#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cstdint>
// duckdb includes
#include "duckdb/main/connection.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
// my includes
#include "logger.hpp"
#include "utils.hpp"
#include "enums/data_type.hpp"
#include "db/column.hpp"
#include "db/table.hpp"


// foreach other tables get the Forigen Key that I reference this table
void Table::setupForeignKeys(const std::unordered_map<std::string, Table *>& tables) {
    for (const auto& [tableName, table] : tables) {
        if (tableName == name) continue;

        std::vector<std::pair<std::shared_ptr<Column>, std::shared_ptr<Column>>> tableLinks;
        tableLinks.reserve(table->pkColumns.size());

        for (const auto& [pkName, pkCol] : table->pkColumns) {
            std::string cleanPkName = generateFkName(tableName, pkName);
            auto it = columns.find(cleanPkName);
            if (it != columns.end()) {
                tableLinks.emplace_back(pkCol, it->second);
            }
        }

        if (tableLinks.size() == table->pkColumns.size()) {
            fkColumns[table] = std::move(tableLinks);
            LOG_INFO(logger_, "Detected Foreign Key Relationship where {} references {}", name, tableName);
        }
    }
}


void Table::inferDBSchema() {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERROR(logger_, "Can't open CSV file at path {}", path);
        throw std::runtime_error("Can't open CSV file at path: " + path);
    }

    // Read first line (header) and one line to deduce float or int
    std::string header, line;
    std::getline(file, header);
    std::getline(file, line);

    file.close();

    // Split line by commas
    std::vector<std::string> columnNames;
    std::vector<std::string> lineValues;
    std::stringstream ssHeader(header), ssLine(line);
    std::string column, lineV;

    while (std::getline(ssHeader, column, ',') && std::getline(ssLine, lineV, ',')) {
        // Remove quotes if present
        column.erase(std::remove(column.begin(), column.end(), '"'), column.end());
        lineV.erase(std::remove(lineV.begin(), lineV.end(), '"'), lineV.end());
        
        columnNames.push_back(column);
        lineValues.push_back(lineV);
    }

    if (columnNames.empty()) {
        LOG_ERROR(logger_, "No columns found in CSV header at path {}", path);
        throw std::runtime_error("No columns found in CSV header: " + path);
    }

    // Initialize schema
    numCols = columnNames.size();
    columns.reserve(numCols);
    columnsType.reserve(numCols);
    columnsOrdered.reserve(numCols);
    csvNameColumn.reserve(numCols);
    rowBytes = 0;

    // Process each column name
    uint16_t idx = 0;
    for (const auto& name : columnNames) {
        // Determine type and primary key
        std::string baseName = name;
        // Trim whitespace
        baseName.erase(0, baseName.find_first_not_of(" \t"));
        baseName.erase(baseName.find_last_not_of(" \t") + 1);

        bool isPk = baseName.ends_with("(P)");
        DataType myType;

        // Remove (P) if present
        if (isPk) {
            baseName = baseName.substr(0, baseName.length() - 3);
            baseName.erase(baseName.find_last_not_of(" \t") + 1); // Trim trailing whitespace
        }

        // Check type suffix
        if (baseName.ends_with("(N)")) {
            lineValues[idx].erase(0, lineValues[idx].find_first_not_of(" \t"));
            lineValues[idx].erase(lineValues[idx].find_last_not_of(" \t") + 1);
            if(lineValues[idx].find_first_of(".") != std::string::npos) {
                myType = DataType::FLOAT;
            } else {
                myType = DataType::INT;
            }
            baseName = baseName.substr(0, baseName.length() - 3);
        } else if (baseName.ends_with("(D)")) {
            myType = DataType::DATETIME;
            baseName = baseName.substr(0, baseName.length() - 3);
        } else if (baseName.ends_with("(T)")) {
            myType = DataType::STRING;
            baseName = baseName.substr(0, baseName.length() - 3);
        } else {
            LOG_WARNING(logger_, "Column '{}' has no type suffix, defaulting to STRING", name);
            myType = DataType::STRING;
        }

        // Trim baseName
        baseName.erase(baseName.find_last_not_of(" \t") + 1);

        if (baseName.empty()) {
            LOG_ERROR(logger_, "Column name '{}' is empty after parsing", name);
            throw std::runtime_error("Invalid column name in CSV header: " + name);
        }

        LOG_DEBUG(logger_, "Column: name='{}', type={}, isPk={}", baseName, static_cast<int>(myType), isPk);

        // Add to schema
        // Add to schema
        auto [it, inserted] = columns.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(baseName),
            std::forward_as_tuple(std::make_shared<Column>(baseName, myType, isPk))
        );

        if (isPk) pkColumns.emplace(baseName, it->second); // Fixed: Use it->second directly
        columnsOrdered.emplace_back(it->second);
        columnsType.emplace_back(myType);
        csvNameColumn.emplace_back(name);
        rowBytes += getDataTypeNumBytes(myType);

        idx++;
    }

}


