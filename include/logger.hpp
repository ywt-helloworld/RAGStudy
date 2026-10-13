#pragma once

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

enum class LogLevel { kInfo, kWarning, kError };

class Logger {
 public:
  // 自动创建父目录，追加写入已有日志。
  static void Init(const std::string& path);
  static void Log(LogLevel level, const std::string& message,
                  bool console = false) noexcept;

  template <typename... Args>
  static void Info(const Args&... args) noexcept {
    Write(LogLevel::kInfo, false, args...);
  }

  template <typename... Args>
  static void Warning(const Args&... args) noexcept {
    Write(LogLevel::kWarning, true, args...);
  }

  template <typename... Args>
  static void Error(const Args&... args) noexcept {
    Write(LogLevel::kError, true, args...);
  }

 private:
  Logger() = default;
  static Logger& Instance();

  template <typename... Args>
  static void Write(LogLevel level, bool console,
                    const Args&... args) noexcept {
    try {
      std::ostringstream message;
      (message << ... << args);
      Log(level, message.str(), console);
    } catch (...) {
      // 日志失败不改变业务流程，也不让异常穿过模型库的回调。
    }
  }

  std::ofstream output_;
  std::mutex mutex_;
};
