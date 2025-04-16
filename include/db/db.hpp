#ifndef YALLASQL_INCLUDE_DB_HPP
#define YALLASQL_INCLUDE_DB_HPP

#include<string>
#include <mutex>
#include<vector>
#include <ranges>
#include<map>

class DB {

private:

    static DB* db_;
    std::string path_;
    static std::mutex mutex_;
    std::map<std::string, std::string> tablesPaths_;

    // private singelton constructor
    DB(const std::string path): path_(path) { refreshTables(); }
    // reload tablesPaths_
    void refreshTables();

public:
    // db can't be cloned
    DB(const DB &) = delete;
    // db can't be assigned
    void operator=(const DB &) = delete;
    // control access to db instance
    static DB *getInstance(const std::string& path);
    // get path of database
    std::string path() const { return path_; }

    ~DB();
};

#endif // YALLASQL_INCLUDE_DB_HPP