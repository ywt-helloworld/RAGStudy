#include "Logger.hpp"

#include <iostream>


namespace {

std::string levelToString(LogLevel level) {
    switch(level) {
        case LogLevel::INFO:
            return "INFO";

        case LogLevel::WARNING:
            return "WARNING";

        case LogLevel::ERROR:
            return "ERROR";
    }

    return "UNKNOWN";
}

} // namespace


void Logger::log(LogLevel level, const std::string& message) {
    std::cout << "[" << levelToString(level) << "] " << message << "\n";
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO,message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING,message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR,message);
}