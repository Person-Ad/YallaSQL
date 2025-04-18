#ifndef YALLASQL_INCLUDE_CLI_SHELL_HPP
#define YALLASQL_INCLUDE_CLI_SHELL_HPP

#include <string>
#include <vector>
#include <replxx.hxx>
#include "command.hpp"
#include "engine/query_engine.hpp"

using Replxx = replxx::Replxx;

/**
 * @class YallaSQLShell
 * @brief A command-line interface shell for interacting with the YallaSQL database.
 */
class YallaSQLShell {
private:
    Replxx rx;
    QueryEngine engine;
    bool running = true; 
    std::vector<Command> commands; 
    std::string current_db = "default"; 
    std::string shell_history = "./logs/yallasql_history.txt";
    std::string input_buffer;
    std::vector<std::string> keywords = {
        "SELECT", "FROM", "WHERE", "AND", "OR", 
        "COUNT", "MIN", "MAX", "SUM", "AVG",
        "DATABASE", "AS", "ORDER BY",
        "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "FULL", "ASC", "DESC",
        "DESCRIBE", " PRAGMA", "table_info"
    };
public:
    /**
     * @brief Constructor to initialize the shell.
     */
    YallaSQLShell();

    /**
     * @brief Destructor to save history of shell.
     */
    ~YallaSQLShell();

    /**
     * @brief Starts the main loop of the shell.
     */
    void run();



private:
    /**
     * @brief Initializes the list of available commands.
     */
    void initializeCommands();

    /**
     * @brief Configures the Replxx library for input handling.
     */
    void setupReplxx();
    
    /**
     * @brief Provides command completions for Replxx.
     * @return A list of possible completions.
     */
    Replxx::completions_t completeCommand(const std::string& input, int& context_len);

    /**
     * @brief Provides hints for commands in Replxx.
     * @return A list of possible hints.
     */
    Replxx::hints_t hintCommand(const std::string& input, int& context_len, Replxx::Color& color);

    /**
     * @brief Highlights syntax in the input for Replxx.
     * @param input The input string.
     * @param colors The colors to apply to the input.
     */
    void highlightSyntax(const std::string& input, Replxx::colors_t& colors);

    /**
     * @brief Clears the terminal screen.
     */
    void clearScreen();

    /**
     * @brief Displays help information for commands.
     * @param args The command to display help for (optional).
     */
    void showHelp(const std::string& args);

    /**
     * @brief Executes an SQL query.
     * @param sql The SQL query to execute.
     */
    void executeSQL(const std::string& sql);

    /**
     * @brief Processes user input from the shell.
     * @param input The input string.
     */
    void processInput(const std::string& input);

    /**
     * @brief Generates the shell prompt string.
     */
    std::string getPrompt();

    /**
     * @brief Displays an animated welcome message.
     */
    void animateWelcome();

    /**
     * @brief Prints a gradient title banner.
     */
    void printGradientTitle();
    /**
     * @brief is ending with semicolumn or not
     */
    bool isCommandComplete(const std::string& line); 
};

#endif // YALLASQL_INCLUDE_CLI_SHELL_HPP