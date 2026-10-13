#pragma once

#include <string>
#include <vector>

#include "chunk.hpp"
#include "config.hpp"
#include "llamamodel.hpp"

class EmbeddingEngine : public LlamaModelBase {
 public:
  explicit EmbeddingEngine(const AppConfig& config);

  // 为单段文本生成 embedding 向量
  std::vector<float> generateEmbedding(const std::string& text);

  // 为一批 Chunk 填充 embedding
  void embedChunks(std::vector<Chunk>& chunks);

 private:
  // 清理输入中残缺或非法的 UTF-8 字节
  std::string sanitizeUtf8(const std::string& text) const;
};