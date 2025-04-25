#pragma once
#include "config.hpp"
#include "duckdb/common/types.hpp"
enum class DataType : __uint8_t {
    INT,
    FLOAT,
    DATETIME,
    STRING
};

[[nodiscard]] inline unsigned int getDataTypeNumBytes(DataType type) {
    if (type == DataType::INT) return sizeof(int); // 32-bit
    if (type == DataType::FLOAT) return sizeof(float); // 32-bit
    if (type == DataType::DATETIME) return sizeof(int64_t); // 64-bit timestamp //TODO: recheck it
    if (type == DataType::STRING) return YallaSQL::MAX_STR_LEN; // 1 char = 1 byte
    return 0; // Fallback for unhandled cases
}

[[nodiscard]] inline DataType getDataTypeFromDuck(duckdb::LogicalType duckType) {
    // handle intervals as
    if(duckType.IsIntegral()) return DataType::INT;
    if(duckType.IsNumeric()) return DataType::FLOAT;
    if(duckType.IsTemporal() || duckType == duckdb::LogicalType::INTERVAL) return DataType::DATETIME;
    return DataType::STRING;

}

[[nodiscard]] inline int64_t getDateTime(const std::string& valueStr) {
    std::tm valueTm = {};
    // Initialize all fields to avoid garbage values
    valueTm.tm_hour = 0;
    valueTm.tm_min = 0;
    valueTm.tm_sec = 0;
    valueTm.tm_year = 0;
    valueTm.tm_mon = 0;
    valueTm.tm_mday = 0;
    
    // Parse the time string
    if (strptime(valueStr.c_str(), "%Y-%m-%d %H:%M:%S", &valueTm) == nullptr) {
        // Handle parsing error
        return 0;
    }
    
    // Convert to time_t (seconds since epoch)
    time_t timeValue = timegm(&valueTm); // Use timegm for UTC or mktime for local time
    
    return static_cast<int64_t>(timeValue);
}

[[nodiscard]] inline std::string getDateTimeStr(int64_t seconds) {
    std::time_t raw_time = static_cast<std::time_t>(seconds);
    
    // Get time structure (use gmtime for UTC or localtime for local timezone)
    std::tm tm_info;
    gmtime_r(&raw_time, &tm_info); // Thread-safe version
    
    char buffer[26];
    // Format the time (strftime is more reliable than put_time)
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", &tm_info);
    
    return std::string(buffer);
}