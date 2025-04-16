#include<thread>
#include<iomanip>
#include<iostream>
#include<vector>
#include<string>
#include<chrono>

// Extended ANSI color codes for rich UI
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
    std::string rgb(int r, int g, int b) {
        return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
    }
    std::string bg_rgb(int r, int g, int b) {
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

namespace UI {
    /**
     * @brief Generates the shell prompt string.
     */
    std::string getPrompt() {
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

    /**
     * @brief Displays an animated welcome message.
     */
    void animateWelcome() {
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

    /**
     * @brief Prints a gradient title banner.
     */
    void printGradientTitle() {
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
               ██║   ██║  ██║███████╗███████╗██║  ██║    ███████║╚██████╔╝██  ███████╗
               ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝══╝  ╚══════╝
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
}