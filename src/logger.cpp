#include "logger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

std::string NormalizeUtf8(const std::string& text) {
  std::string result;
  result.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i]);
    size_t length = first < 0x80                     ? 1
                    : first >= 0xc2 && first <= 0xdf ? 2
                    : first >= 0xe0 && first <= 0xef ? 3
                    : first >= 0xf0 && first <= 0xf4 ? 4
                                                     : 0;
    bool valid = length != 0 && length <= text.size() - i;
    for (size_t j = 1; valid && j < length; ++j) {
      const auto next = static_cast<unsigned char>(text[i + j]);
      valid = next >= 0x80 && next <= 0xbf;
      if (j == 1) {
        valid = valid && !(first == 0xe0 && next < 0xa0) &&
                !(first == 0xed && next >= 0xa0) &&
                !(first == 0xf0 && next < 0x90) &&
                !(first == 0xf4 && next >= 0x90);
      }
    }
    if (valid) {
      result.append(text, i, length);
      i += length;
    } else {
      // 异常消息可能包含截断的 UTF-8 字节，替换后保证日志可读。
      result.append("\xef\xbf\xbd");
      ++i;
    }
  }
  return result;
}

}  // namespace

Logger& Logger::Instance() {
  static Logger logger;
  return logger;
}

void Logger::Init(const std::string& path) {
  const auto file = std::filesystem::u8path(path);
  if (file.has_parent_path()) {
    std::filesystem::create_directories(file.parent_path());
  }
  std::ofstream output(file, std::ios::binary | std::ios::app);
  if (!output) {
    throw std::runtime_error("无法打开日志文件: " + path);
  }
  auto& logger = Instance();
  std::lock_guard<std::mutex> lock(logger.mutex_);
  logger.output_ = std::move(output);
}

void Logger::Log(LogLevel level, const std::string& message,
                 bool console) noexcept {
  try {
    auto& logger = Instance();
    std::lock_guard<std::mutex> lock(logger.mutex_);
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch())
            .count() %
        1000;
    const char* name = level == LogLevel::kError     ? "ERROR"
                       : level == LogLevel::kWarning ? "WARN"
                                                     : "INFO";
    std::ostringstream prefix;
    prefix << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.'
           << std::setfill('0') << std::setw(3) << milliseconds << "] [" << name
           << "] ";

    std::istringstream lines(NormalizeUtf8(message));
    std::string line;
    while (std::getline(lines, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) {
        continue;
      }
      if (logger.output_.is_open()) {
        logger.output_ << prefix.str() << line << '\n';
      }
      if (console) {
        std::cerr << '[' << name << "] " << line << '\n';
      }
    }
    if (logger.output_.is_open()) {
      logger.output_.flush();
      if (!logger.output_) {
        logger.output_.close();
        std::cerr << "[Logger] 写入日志失败，已停止文件记录。\n";
      }
    }
  } catch (...) {
    // 日志写入不向业务代码抛出异常。
  }
}
