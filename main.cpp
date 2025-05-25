#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "engine/query_engine.hpp"


// Helper function to extract the base filename without path or extension
std::string getBaseFilename(const std::string& filepath) {
    // Find the last path separator ( '/' or '\' )
    size_t lastSlash = filepath.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);

    // Remove the extension (e.g., .txt)
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        filename = filename.substr(0, lastDot);
    }

    return filename;
}

int main(int argc, char** argv) {
    // Check if the correct number of arguments is provided
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <folder_name> <query.txt>\n";
        return 1;
    }
    std::cout << "okay starting...\n";

    // Extract folder_name and query file path from command-line arguments
    std::string folder_name = argv[1];
    std::string query_file = argv[2];

    QueryEngine engine;
    // Set the dataset using "use folder_name"
    std::string use_command = "use " + folder_name;
    engine.execute(use_command);

    // Read queries from query.txt
    std::ifstream file(query_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << query_file << "\n";
        return 1;
    }

    // Read the file content
    std::string query = "";
    std::string line;
    while (std::getline(file, line)) {
        query += line + "\n";
    }
    // Get the base filename (e.g., "query1" from "./query1.txt")
    std::string base_filename = getBaseFilename(query_file);

    // Construct the output filename (e.g., "Team10_query1.csv")
    std::string output_filename = "Team10_" + base_filename + ".csv";

    std::cout << "start executing: " << query << "\n";
    engine.execute(query, output_filename);
}
