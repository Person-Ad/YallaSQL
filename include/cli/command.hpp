#include <string>
#include <replxx.hxx>

// command structure with descriptions for better organization
struct Command {
    std::string name;
    std::string description;
    std::string syntax;
    std::function<void(const std::string&)> handler;
};
