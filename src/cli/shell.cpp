#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cctype>

#include "cli/shell.hpp"
#include "cli/ui.hpp"

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
        
        {"USE", "  Load DB from Folder", "USE <dirname>"},
        {"DESCRIBE", "  List tables with summary", "DESCRIBE"},
        {"SELECT", "  Query data from tables", "SELECT ... FROM ... [WHERE ...]"},
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
    rx.history_load("YallaSQLShell_history.txt");
    
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

    for (const auto& cmd : commands) {
        if (cmd.name.find(input) == 0) {
            completions.emplace_back(cmd.name + " ");
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
    UI::printGradientTitle();
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
        engine.execute(input);
    }
}

void YallaSQLShell::run() {
    // Setup the terminal
    std::cout << "\033[2J\033[H";  // Clear screen
    std::cout << Cursor::BLINKING_BAR;
    
    // Print welcome banner
    UI::printGradientTitle();
    UI::animateWelcome();
    
    // Main input loop
    while (running) {
        // Get input with our beautiful prompt
        const char* input = rx.input(UI::getPrompt());
        
        if (input == nullptr) {
            // Ctrl+D pressed
            std::cout << "Goodbye!\n";
            break;
        }
        
        std::string line(input);
        if (!line.empty()) {
            rx.history_add(line);
            rx.history_save("yallasql_history.txt");
            processInput(line);
        }
    }
    
    // Clean up terminal settings
    std::cout << "\033[?1000l";
}

