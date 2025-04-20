#pragma once
#include "config.hpp"

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

[[nodiscard]] inline int64_t getDateTime(const std::string& valueStr) {
    std::tm valueTm = {};
    strptime(valueStr.c_str(), "%Y-%m-%d %H:%M:%S", &valueTm); //* format may vary
    return std::mktime(&valueTm); // stores seconds since epoch
}


[[nodiscard]] inline std::string getDateTimeStr(int64_t seconds) {
    std::time_t raw_time = static_cast<std::time_t>(seconds);
    std::tm* tm_ptr = std::localtime(&raw_time);

    std::ostringstream oss;
    oss << std::put_time(tm_ptr, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}