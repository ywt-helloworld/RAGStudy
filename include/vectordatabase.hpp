#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "chunk.hpp"

struct SearchResult {
  const Chunk* chunk;
  float score;
};

class VectorDatabase {
 public:
  VectorDatabase() = default;
  explicit VectorDatabase(const std::string& dbPath);

  // 插入单个 Chunk 或批量插入 Chunk
  void insert(const Chunk& chunk);
  void insert(const std::vector<Chunk>& newChunks);

  // 返回与 queryEmbedding 最相似的 topK 个 Chunk
  std::vector<SearchResult> search(const std::vector<float>& queryEmbedding,
                                   std::size_t topK) const;

  // 混合检索：真实向量相似度为主，关键词命中为辅
  std::vector<SearchResult> hybridSearch(
      const std::string& query, const std::vector<float>& queryEmbedding,
      std::size_t topK) const;

  // 将整个向量数据库保存到磁盘 / 从磁盘加载
  bool saveToDisk(const std::string& path) const;
  bool loadFromDisk(const std::string& path);

  void clear();
  std::size_t size() const;
  bool empty() const;
  const std::vector<Chunk>& documents() const { return chunks; }

 private:
  static constexpr std::uint32_t MAGIC = 0x4D524147;
  static constexpr std::uint32_t VERSION = 2;

  std::vector<Chunk> chunks;

  // 这个函数用于检测语义相似性
  static float cosineSimilarity(const std::vector<float>& a,
                                const std::vector<float>& b);

  // 这个函数用于检测关键词重合程度
  static float keyWords(const std::string& query, const Chunk& chunk);
};
