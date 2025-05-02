#include<thread>
#include<iomanip>
#include<iostream>
#include<vector>
#include<string>
#include<chrono>
#include "quill/LogMacros.h"
// Extended ANSI color codes for rich UI

#ifndef YALLASQL_UTILS_HPP
#define YALLASQL_UTILS_HPP
namespace YallaSQL::UTILS {

namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string DIM = "\033[2m";
    const std::string ITALIC = "\033[3m";
    const std::string UNDERLINE = "\033[4m";
    
    // Foreground colors
    const std::string BLACK = "\033[30m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    
    // Background colors
    const std::string BG_BLACK = "\033[40m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_BLUE = "\033[44m";
    const std::string BG_MAGENTA = "\033[45m";
    const std::string BG_CYAN = "\033[46m";
    const std::string BG_WHITE = "\033[47m";
    
    // 256-color support
    inline std::string rgb(int r, int g, int b) {
        return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }
    inline std::string bg_rgb(int r, int g, int b) {
        return "\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }
}


namespace Cursor {
    const std::string BLINKING_BLOCK = "\033[1 q";
    const std::string STEADY_BLOCK = "\033[2 q";
    const std::string BLINKING_UNDERLINE = "\033[3 q";
    const std::string STEADY_UNDERLINE = "\033[4 q";
    const std::string BLINKING_BAR = "\033[5 q";      // Most modern terminals
    const std::string STEADY_BAR = "\033[6 q";        // Default in many terminals
    const std::string HIDE = "\033[0 q" "\033[?25l";  // Hide cursor
    const std::string SHOW = "\033[?25h";             // Show cursor
}


#define MEASURE_EXECUTION_TIME(label, code_block)                                         \
{                                                                                         \
    auto start = std::chrono::high_resolution_clock::now();                               \
    code_block;                                                                           \
    auto end = std::chrono::high_resolution_clock::now();                                 \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);   \
    std::cout << Color::BOLD << Color::MAGENTA << "<⏱️  " << label << " > "                \
    << Color::RESET << Color::CYAN << duration.count() << " ms "                         \
    << Color::RESET << std::endl;                                                        \
}

#define MEASURE_EXECUTION_TIME_LOGGER(logger, label, code_block)                          \
{                                                                                         \
    auto start = std::chrono::high_resolution_clock::now();                               \
    code_block;                                                                           \
    auto end = std::chrono::high_resolution_clock::now();                                 \
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);   \
    std::cout << Color::BOLD << Color::MAGENTA << "<⏱️  " << label << " > "                  \
    << Color::RESET << Color::CYAN << duration.count() << " ms "                            \
    << Color::RESET << std::endl;                                                           \
    LOG_INFO(logger, "{} took {} ms", std::string_view{label}, duration.count());           \
}

#define MEASURE_EXECUTION_TIME_MICRO_LOGGER(logger, label, code_block)                          \
{                                                                                         \
    auto start = std::chrono::high_resolution_clock::now();                               \
    code_block;                                                                           \
    auto end = std::chrono::high_resolution_clock::now();                                 \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);   \
    std::cout << Color::BOLD << Color::MAGENTA << "<⏱️  " << label << "> "                  \
    << Color::RESET << Color::CYAN << duration.count() << " μs "                            \
    << Color::RESET << std::endl;                                                           \
    LOG_INFO(logger, "{} took {} μs", std::string_view{label}, duration.count());           \
}

}
#endif
