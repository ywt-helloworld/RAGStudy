#pragma once

#include <string>
#include <vector>

#include "config.hpp"
#include "llamamodel.hpp"
#include "vectordatabase.hpp"

struct GenerationResult {
  std::string text;
  std::string error;
  std::string prompt;
};

class GenerationEngine : public LlamaModelBase {
 public:
  explicit GenerationEngine(const AppConfig& config);

  // 根据用户问题和检索结果流式生成回答
  void generateStream(const std::string& query,
                      const std::vector<SearchResult>& results);

  // 批量评测捕获原始输出；模型调用错误保留在 error 中。
  GenerationResult Generate(const std::string& query,
                            const std::vector<SearchResult>& results,
                            bool stream = false, bool single_choice = false);

 private:
  // 将检索到的 Chunk 拼接成带出处的上下文
  std::string buildContext(const std::vector<SearchResult>& results) const;

  int maxOutputTokens;
  double temperature;
  int samplerTopK;
  double topP;

  std::string genModelPath;
};
