#include "vectordatabase.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "logger.hpp"

namespace {

// 判断一个 UTF-8 字符大概占几个字节
std::size_t utf8CharLength(unsigned char c) {
  if ((c & 0x80U) == 0) {
    return 1;
  }
  if ((c & 0xE0U) == 0xC0U) {
    return 2;
  }
  if ((c & 0xF0U) == 0xE0U) {
    return 3;
  }
  if ((c & 0xF8U) == 0xF0U) {
    return 4;
  }
  return 1;
}

// 把字符串按 UTF-8 字符切开
std::vector<std::string> splitUtf8Chars(const std::string& text) {
  std::vector<std::string> chars;

  for (std::size_t i = 0; i < text.size();) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    std::size_t len = utf8CharLength(c);

    if (i + len > text.size()) {
      len = 1;
    }

    chars.push_back(text.substr(i, len));
    i += len;
  }

  return chars;
}

// 这些字对检索帮助不大，提取关键词时不考虑这些词
bool isStopWord(const std::string& token) {
  static const std::unordered_set<std::string> stopWords = {
      "的", "了", "和", "与", "及", "或", "在", "是", "为", "被", "把",
      "这", "那", "其", "他", "她", "它", "们", "个", "一", "有", "无",
      "何", "如", "么", "什", "哪", "些", "多", "少", "第", "次", "中",
      "上", "下", "里", "后", "前", "时", "于", "之", "所", "以", "吗",
      "呢", "谁", "从", "到", "并", "而", "又", "也", "都", "共", "问",
      "答", "说", "曰", "可", "不", "让", "给", "将", "使", "？", "?",
      "，", ",",  "。", ".",  "；", ";",  "：", ":",  "！", "!",  "、",
      "\"", "'",  "“",  "”",  "‘",  "’",  "（", "）", "(",  ")",  "《",
      "》", "[",  "]",  "【", "】", "\n", "\r", "\t", " "};

  return stopWords.find(token) != stopWords.end();
}

// 从问题中提取关键词
// 例如：“刘备和谁结义？”
// 这个字符串可能得到：刘、备、结、义、刘备、备结、结义、刘备结、备结义
std::vector<std::string> extractQuestionKeywords(const std::string& query) {
  // 按 UTF-8 字符切开
  const std::vector<std::string> chars = splitUtf8Chars(query);

  // 去掉停用词和标点
  std::vector<std::string> usefulChars;
  std::vector<std::string> keywords;
  std::string asciiWord;

  auto flushAsciiWord = [&]() {
    if (asciiWord.size() >= 2) {
      keywords.push_back(asciiWord);
    }
    asciiWord.clear();
  };

  for (const std::string& ch : chars) {
    if (ch.size() == 1 &&
        std::isalnum(static_cast<unsigned char>(ch[0])) != 0) {
      asciiWord.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(ch[0]))));
      continue;
    }

    flushAsciiWord();

    if (isStopWord(ch)) {
      continue;
    }

    usefulChars.push_back(ch);
    keywords.push_back(ch);
  }

  flushAsciiWord();

  // 中文单字匹配太弱，所以补充二字词
  for (std::size_t i = 0; i + 1 < usefulChars.size(); ++i) {
    keywords.push_back(usefulChars[i] + usefulChars[i + 1]);
  }

  // 再补充三字词，用来保留人名，如“诸葛亮”这类信息
  for (std::size_t i = 0; i + 2 < usefulChars.size(); ++i) {
    keywords.push_back(usefulChars[i] + usefulChars[i + 1] +
                       usefulChars[i + 2]);
  }

  // 去重，避免同一个关键词重复被截取
  std::set<std::string> seen;
  std::vector<std::string> uniqueKeywords;

  for (const std::string& word : keywords) {
    if (seen.insert(word).second) {
      uniqueKeywords.push_back(word);
    }
  }

  return uniqueKeywords;
}

}  // namespace

VectorDatabase::VectorDatabase(const std::string& dbPath) {
  loadFromDisk(dbPath);
}

void VectorDatabase::insert(const Chunk& chunk) { chunks.push_back(chunk); }

void VectorDatabase::insert(const std::vector<Chunk>& newChunks) {
  for (const auto& chunk : newChunks) {
    insert(chunk);
  }
}

std::vector<SearchResult> VectorDatabase::search(
    const std::vector<float>& queryEmbedding, std::size_t topK) const {
  std::vector<SearchResult> results;

  if (queryEmbedding.empty() || chunks.empty() || topK == 0) {
    return results;
  }

  struct Compare {
    bool operator()(const SearchResult& a, const SearchResult& b) const {
      return a.score > b.score;
    }
  };

  std::priority_queue<SearchResult, std::vector<SearchResult>, Compare> minHeap;

  for (const auto& chunk : chunks) {
    if (chunk.embedding.empty()) {
      continue;
    }

    if (chunk.embedding.size() != queryEmbedding.size()) {
      continue;
    }

    float score = cosineSimilarity(queryEmbedding, chunk.embedding);

    SearchResult result{&chunk, score};

    if (minHeap.size() < topK) {
      minHeap.push(result);
    } else if (score > minHeap.top().score) {
      minHeap.pop();
      minHeap.push(result);
    }
  }

  while (!minHeap.empty()) {
    results.push_back(minHeap.top());
    minHeap.pop();
  }

  std::reverse(results.begin(), results.end());

  return results;
}

// 最终分数 = 向量相似度 + 关键词得分 *
// 0.35（这个数据以实际测试为准，根据不同书本应该不同）
std::vector<SearchResult> VectorDatabase::hybridSearch(
    const std::string& query, const std::vector<float>& queryEmbedding,
    std::size_t topK) const {
  std::vector<SearchResult> results;

  if (queryEmbedding.empty() || chunks.empty() || topK == 0) {
    return results;
  }

  struct Compare {
    bool operator()(const SearchResult& a, const SearchResult& b) const {
      return a.score > b.score;
    }
  };

  std::priority_queue<SearchResult, std::vector<SearchResult>, Compare> minHeap;

  for (const auto& chunk : chunks) {
    if (chunk.embedding.empty()) {
      continue;
    }

    if (chunk.embedding.size() != queryEmbedding.size()) {
      continue;
    }

    const float vectorScore = cosineSimilarity(queryEmbedding, chunk.embedding);
    const float keywordScore = keyWords(query, chunk);

    // 向量相似度仍然是主力，关键词只是辅助
    const float finalScore = vectorScore + keywordScore * 0.35f;

    SearchResult result{&chunk, finalScore};

    if (minHeap.size() < topK) {
      minHeap.push(result);
    } else if (finalScore > minHeap.top().score) {
      minHeap.pop();
      minHeap.push(result);
    }
  }

  while (!minHeap.empty()) {
    results.push_back(minHeap.top());
    minHeap.pop();
  }

  std::reverse(results.begin(), results.end());

  return results;
}

bool VectorDatabase::saveToDisk(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);

  if (!out.is_open()) {
    Logger::Error("Failed to open vector database for saving: ", path);
    return false;
  }

  std::uint32_t magic = MAGIC;
  std::uint32_t version = VERSION;
  std::uint64_t count = static_cast<std::uint64_t>(chunks.size());

  out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

  out.write(reinterpret_cast<const char*>(&version), sizeof(version));

  out.write(reinterpret_cast<const char*>(&count), sizeof(count));

  if (!out) {
    Logger::Error("Failed to write chunk count to vector database: ", path);
    return false;
  }

  for (std::uint64_t i = 0; i < count; ++i) {
    try {
      chunks[static_cast<std::size_t>(i)].serialize(out);
    } catch (const std::exception& e) {
      Logger::Error("Failed to write chunk ", i,
                    " to vector database: ", e.what());
      return false;
    }
  }

  out.close();

  if (!out) {
    Logger::Error("Failed to close vector database file after saving: ", path);
    return false;
  }

  Logger::Info("数据库保存完成: ", path, "，文本块数=", chunks.size());
  return true;
}

bool VectorDatabase::loadFromDisk(const std::string& path) {
  std::ifstream in(path, std::ios::binary);

  if (!in.is_open()) {
    Logger::Error("Failed to open vector database for loading: ", path);
    return false;
  }

  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t count = 0;

  in.read(reinterpret_cast<char*>(&magic), sizeof(magic));

  in.read(reinterpret_cast<char*>(&version), sizeof(version));

  in.read(reinterpret_cast<char*>(&count), sizeof(count));

  if (!in) {
    Logger::Error("Failed to read vector database header: ", path);
    return false;
  }

  if (magic != MAGIC) {
    Logger::Error("Invalid vector database magic number: ", path);
    return false;
  }

  if (version != VERSION) {
    Logger::Error("Unsupported vector database version: ", version,
                  ". Rebuild the database with --ingest.");
    return false;
  }

  clear();

  for (std::uint64_t i = 0; i < count; ++i) {
    try {
      Chunk chunk = Chunk::deserialize(in);
      chunks.push_back(std::move(chunk));
    } catch (const std::exception& e) {
      Logger::Error("Failed to read chunk ", i,
                    " from vector database: ", e.what());
      clear();
      return false;
    }
  }

  Logger::Info("数据库加载完成: ", path, "，文本块数=", chunks.size());
  return true;
}

void VectorDatabase::clear() { chunks.clear(); }

std::size_t VectorDatabase::size() const { return chunks.size(); }

bool VectorDatabase::empty() const { return chunks.empty(); }

float VectorDatabase::cosineSimilarity(const std::vector<float>& a,
                                       const std::vector<float>& b) {
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return 0.0f;
  }

  float AB = 0.0f;
  float A2 = 0.0f;
  float B2 = 0.0f;

  for (std::size_t i = 0; i < a.size(); ++i) {
    AB += a[i] * b[i];
    A2 += a[i] * a[i];
    B2 += b[i] * b[i];
  }

  if (A2 == 0.0f || B2 == 0.0f) {
    return 0.0f;
  }

  return AB / (std::sqrt(A2) * std::sqrt(B2));
}

// 新增：关键词得分
float VectorDatabase::keyWords(const std::string& query, const Chunk& chunk) {
  const std::vector<std::string> keywords = extractQuestionKeywords(query);

  if (keywords.empty()) {
    return 0.0f;
  }

  const std::string haystack = chunk.metadata + "\n" + chunk.text;

  float totalWeight = 0.0f;
  float matchedWeight = 0.0f;

  for (const std::string& word : keywords) {
    float weight = 0.08f;

    // 二字词、三字词比单字更有信息量，所以权重大一点
    if (word.size() >= 9) {
      weight = 0.45f;
    } else if (word.size() >= 6) {
      weight = 0.22f;
    } else if (word.size() >= 3) {
      weight = 0.08f;
    } else if (word.size() >= 2) {
      weight = 0.16f;
    }

    totalWeight += weight;

    if (haystack.find(word) != std::string::npos) {
      matchedWeight += weight;
    }
  }

  if (totalWeight <= 0.0f) {
    return 0.0f;
  }

  float score = matchedWeight / totalWeight;

  if (score > 1.0f) {
    score = 1.0f;
  }

  return score;
}
