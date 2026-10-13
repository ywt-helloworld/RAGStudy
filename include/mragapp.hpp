#pragma once

#include <string>
#include <vector>

#include "config.hpp"

class MragApp {
 public:
  explicit MragApp(const AppConfig& config_);

  // 离线建库：直接读取 JSON 文档记录，生成向量，保存到 .db 文件。
  bool buildKnowledgeBase(const std::string& jsonPath,
                          const std::string& dbPath);

  // 多个 JSON 文件合并建库。
  bool buildKnowledgeBase(const std::vector<std::string>& jsonPaths,
                          const std::string& dbPath);

  // 在线问答：加载数据库，循环读取问题，检索并生成回答
  bool chatLoop(const std::string& dbPath);

  // 单选题评测，逐题写入项目内 test/result。
  bool RunTest(const std::string& questions_path, const std::string& db_path);

 private:
  struct BackendGuard {
    BackendGuard();
    ~BackendGuard();

    BackendGuard(const BackendGuard& other) = delete;
    BackendGuard& operator=(const BackendGuard& other) = delete;
  };

  BackendGuard backend;
  AppConfig config;
};
