#pragma once
#include <string>
#include <cstdint>

#include "enums/data_type.hpp"

struct Column {
    std::string name;
    DataType type : 2;    // 2 bits for DataType
    bool isPk : 1;        // 1 bit for primary key
    bool isFk : 1;        // 1 bit for foreign key
    uint8_t padding : 4;  // Explicit padding to align to byte boundary
    unsigned int bytes;
    
    Column(std::string name_, DataType t, bool pk = false, bool fk = false)
        : name(std::move(name_)), type(t), isPk(pk), isFk(fk), padding(0),  bytes(getDataTypeNumBytes(t)) {}

    // Move constructor
    Column(Column&&) = default;
    Column& operator=(Column&&) = default;

    // Delete copy to prevent unintended copies
    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;
};
