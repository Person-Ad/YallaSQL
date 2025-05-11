#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cctype>

#include "cli/shell.hpp"
#include "utils.hpp"
#include "utils/macros.hpp"

using namespace YallaSQL::UTILS;

using Replxx = replxx::Replxx;


YallaSQLShell::YallaSQLShell() {
    initializeCommands();
    setupReplxx();
}

YallaSQLShell::~YallaSQLShell() {
    rx.history_save(shell_history);
}

void YallaSQLShell::initializeCommands() {
    commands = {
        {"_exit", "Exit the program", ".exit"},
        {"_clear", "  Clear the screen", ".clear"},
        {"_help", "  Show help message", ".help [command]"},
        
        {"duckdb", " Execute statement on DuckDB", "duckdb SELECT ...."},
        {"USE", "  Load DB from Folder", "USE <dirname>"},
        {"DESCRIBE", "  List tables with summary", "DESCRIBE"},
        {"SELECT", "  Query data from tables", "SELECT ... FROM ... [WHERE ...]"},
        {"EXPLAIN", " Explain Query in file", "EXPLAIN...."}
    };
}

void YallaSQLShell::setupReplxx() {
    rx.install_window_change_handler();
    rx.set_max_history_size(1000);
    rx.set_unique_history(true);
    rx.set_complete_on_empty(false);
    rx.set_beep_on_ambiguous_completion(false);
    rx.set_max_hint_rows(4);
    rx.set_hint_delay(1);
    
    // Load history from file
    rx.history_load(shell_history);
    
    // Set up callbacks
    rx.set_completion_callback(
        [this](const std::string& input, int& context_len) -> Replxx::completions_t {
            return completeCommand(input, context_len);
        });
    
    rx.set_hint_callback(
        [this](const std::string& input, int& context_len, Replxx::Color& color) -> Replxx::hints_t {
            return hintCommand(input, context_len, color);
        });

    rx.set_highlighter_callback(
        [this](const std::string& input, Replxx::colors_t& colors) -> void {
            highlightSyntax(input, colors);
        });
}



Replxx::completions_t YallaSQLShell::completeCommand(const std::string& input, int& context_len) {
    Replxx::completions_t completions;
    auto idx = input.find_last_of(' ');

    const std::string lastKeyword = idx == std::string::npos ? input : input.substr(idx + 1);

    for (const auto& cmd : commands) {
        if (cmd.name.find(lastKeyword) == 0) {
            completions.emplace_back(cmd.name + " ");
        }
    }

    for (const auto& keyword : keywords) {
        if (keyword.find(lastKeyword) == 0) {
            completions.emplace_back(keyword + " ");
        }
    }

    return completions;
}

Replxx::hints_t YallaSQLShell::hintCommand(const std::string& input, int& context_len, Replxx::Color& color) {
    Replxx::hints_t hints;
    if(input.length() == 0) return hints;

    for (const auto& cmd : commands) {
        if (cmd.name.find(input) == 0) {
            color = Replxx::Color::YELLOW;
            hints.emplace_back(cmd.name);
            break;
        }
    }

    return hints;
}

void YallaSQLShell::highlightSyntax(const std::string& input, Replxx::colors_t& colors) {
    // SQL keywords (convert to uppercase for case-insensitive matching)
    std::string upper_input = input;
    std::transform(upper_input.begin(), upper_input.end(), upper_input.begin(), ::toupper);
    
    // Highlight SQL keywords
    for (size_t i = 0; i < input.length(); ++i) {
        for (const auto& keyword : keywords) {
            if (i + keyword.length() <= upper_input.length() && 
                upper_input.substr(i, keyword.length()) == keyword) {
                
                // Check if it's a complete word (not part of another word)
                bool is_start = (i == 0 || !std::isalnum(upper_input[i-1]));
                bool is_end = (i + keyword.length() == upper_input.length() || 
                                !std::isalnum(upper_input[i + keyword.length()]));
                
                if (is_start && is_end) {
                    // Highlight the entire keyword
                    for (size_t j = 0; j < keyword.length(); ++j) {
                        colors[i + j] = Replxx::Color::BLUE;
                    }
                }
            }
        }
        
        // Highlight commands starting with '_'
        if (input[i] == '_' && (i == 0 || std::isspace(input[i-1]))) {
            colors[i] = Replxx::Color::YELLOW;
            size_t j = i + 1;
            while (j < input.length() && std::isalpha(input[j])) {
                colors[j] = Replxx::Color::YELLOW;
                ++j;
            }
        }
        
        // Highlight strings
        if (input[i] == '\'' || input[i] == '\"') {
            char quote = input[i];
            colors[i] = Replxx::Color::MAGENTA;
            ++i;
            
            while (i < input.length() && input[i] != quote) {
                colors[i] = Replxx::Color::MAGENTA;
                if (input[i] == '\\' && i + 1 < input.length()) {
                    colors[i+1] = Replxx::Color::MAGENTA;
                    i += 2;
                } else {
                    ++i;
                }
            }
            
            if (i < input.length()) {
                colors[i] = Replxx::Color::MAGENTA;
            }
        }
    }
}

void YallaSQLShell::clearScreen() {
    std::cout << "\033[2J\033[H";
    printGradientTitle();
}

void YallaSQLShell::showHelp(const std::string& input) {
    // load args
    std::string args;
    size_t space_pos = input.find(' ');
    
    if (space_pos != std::string::npos) {
        args = input.substr(space_pos + 1);
    } else {
        args = "";
    }

    if (args.empty()) {
        std::cout << Color::CYAN << Color::BOLD << "Available commands:" << Color::RESET << "\n\n";
        
        for (const auto& cmd : commands) {
            std::cout << "  " << Color::BOLD << Color::YELLOW << std::left << std::setw(12) 
                      << cmd.name << Color::RESET 
                      << " " << cmd.description << "\n";
            std::cout << "      " << Color::DIM << cmd.syntax << Color::RESET << "\n\n";
        }
    } else {
        // Show help for specific command
        for (const auto& cmd : commands) {
            if (cmd.name == args || cmd.name == "_" + args) {
                std::cout << Color::BOLD << Color::YELLOW << cmd.name << Color::RESET 
                          << " - " << cmd.description << "\n";
                std::cout << "Syntax: " << Color::ITALIC << cmd.syntax << Color::RESET << "\n";
                return;
            }
        }
        std::cout << Color::RED << "Unknown command: " << args << Color::RESET << "\n";
    }
}


void YallaSQLShell::processInput(const std::string& input) {
    if (input.empty()) return;

    if(input.find("_help") == 0) {
        showHelp(input);
    } else if (input.find("_clear") == 0) {
        clearScreen();
    } else if (input.find("_exit") == 0) {
        running = false;
    } else {
        try{
            std::string output;
            // MEASURE_EXECUTION_TIME("execute time",
            //     output = engine.execute(input);
            // )
            PROFILING_GPU("execute time", 2, 5,
                output = engine.execute(input);
            );
            std::cout << output << "\n";
        } catch(std::exception &e) {
            std::cout << "UNDEFINE ERROR: " << e.what() << "\n";
        }
    }
}

void YallaSQLShell::run() {
    input_buffer = ""; // buffer to allow multiline
    // Setup the terminal
    std::cout << "\033[2J\033[H";  // Clear screen
    std::cout << Cursor::BLINKING_BAR;
    
    // Print welcome banner
    printGradientTitle();
    animateWelcome();
    
    // Main input loop
    while (running) {
        // Get input with our beautiful prompt
        const char* input = rx.input(input_buffer.empty() ? getPrompt() : Color::DIM + "........................ ");
        
        if (input == nullptr) {
            // Ctrl+D pressed
            std::cout << "Goodbye!\n";
            break;
        }
        std::string line(input);
        input_buffer += input_buffer.empty() ? line : "\n" + line; // add space between lines
        if(isCommandComplete(line)) {
            rx.history_add(input_buffer);
            rx.history_save(shell_history);
            processInput(input_buffer);
            input_buffer = "";
        }
    }
    
    // Clean up terminal settings
    std::cout << "\033[?1000l";
}

bool YallaSQLShell::isCommandComplete(const std::string& line) {
    std::string trimmed = line;
    trimmed.erase(trimmed.find_last_not_of(" \n\r\t") + 1);

    // Check if the line ends with a semicolon or start with command _help, _exist, _clear
    return (!trimmed.empty() && trimmed.back() == ';') || (trimmed[0] == '_');
}

std::string YallaSQLShell::getPrompt() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    
    std::string time_str = std::ctime(&now_time);
    time_str = time_str.substr(11, 5); // Get HH:MM
    
    std::string prompt = Color::bg_rgb(60, 80, 80) + Color::rgb(200, 200, 255) + " " + time_str + "  " +
        Color::RESET +
        Color::bg_rgb(60, 60, 80) + Color::BOLD + Color::rgb(255, 255, 200) + " yallaSQL λ " +
        Color::RESET + " ";
    
    return prompt;
}


void YallaSQLShell::animateWelcome() {
    std::string message = "Welcome to YallaSQL v2.0 - The most Funny CLI database tool";
    for (size_t i = 0; i < message.size(); ++i) {
        std::cout << Color::rgb(100 + i*2, 150 + i, 200 - i) 
                << message[i] << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << Color::RESET << "\n\n";
    
    std::cout << Color::ITALIC << Color::CYAN 
            << "Type " << Color::RESET << Color::YELLOW << ".help" 
            << Color::RESET << Color::CYAN << " for available commands" 
            << Color::RESET << "\n\n";
}


void YallaSQLShell::printGradientTitle() {
    std::vector<std::string> gradient_colors = {
        Color::rgb(255, 0, 102),    // Pink
        Color::rgb(255, 51, 102),
        Color::rgb(255, 102, 102),
        Color::rgb(255, 153, 102),  // Orange
        Color::rgb(255, 204, 102),   // Yellow
        Color::rgb(204, 255, 102),   // Lime
        Color::rgb(102, 255, 102),   // Green
        Color::rgb(102, 255, 204),   // Teal
        Color::rgb(102, 204, 255),   // Light Blue
        Color::rgb(102, 102, 255),    // Blue
        Color::rgb(153, 102, 255),    // Purple
        Color::rgb(204, 102, 255)     // Violet
    };


    std::string title = R"(
██╗   ██╗ █████╗ ██╗     ██╗      █████╗     ███████╗ ██████╗     ██╗     
╚██╗ ██╔╝██╔══██╗██║     ██║     ██╔══██╗    ██╔════╝██╔═══██╗    ██║     
 ╚████╔╝ ███████║██║     ██║     ███████║    ███████╗██║   ██║    ██║     
  ╚██╔╝  ██╔══██║██║     ██║     ██╔══██║    ╚════██║██║   ██║    ██║     
   ██║   ██║  ██║███████╗███████╗██║  ██║    ███████║╚█████████╗  ███████╗
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚════════╝  ╚══════╝
                )";
    std::istringstream iss(title);
    std::string line;
    size_t color_index = 0;
    
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            std::cout << gradient_colors[color_index % gradient_colors.size()] 
                    << line << Color::RESET << "\n";
            color_index++;
        }
    }
}
