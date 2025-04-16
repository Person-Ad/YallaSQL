#ifndef YALLASQL_INCLUDE_ENGINE_HPP
#define YALLASQL_INCLUDE_ENGINE_HPP

#include <stdexcept>
#include <string>
#include "db/db.hpp"

class QueryEngine {

private:
    DB* db_;

    void useDB(const std::string input) {
        std::string path;
        size_t space_pos = input.find(' ');
        
        if (space_pos != std::string::npos)  
            path = input.substr(space_pos + 1);
        else
            throw std::runtime_error("No Direcroty was written");

        DB::setPath(path);
    }

public:
    QueryEngine() {
        db_ = DB::getInstance();
    }

    std::string execute(std::string query) {
        if(query.find("USE") == 0) {
            useDB(query);
        }
        return "success";
    }

    

};

#endif