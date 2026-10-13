#include "embeddingengine.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.hpp"

EmbeddingEngine::EmbeddingEngine(const AppConfig& config)
    : LlamaModelBase(config, ModelMode::Embedding) {}

std::vector<float> EmbeddingEngine::generateEmbedding(const std::string& text) {
  std::string cleanText = sanitizeUtf8(text);

  if (cleanText.empty()) {
    return {};
  }

  std::vector<llama_token> tokens = tokenize(cleanText, true, false);

  if (tokens.empty()) {
    return {};
  }

  std::size_t limit = static_cast<std::size_t>(std::min(nCtx, nBatch));

  if (limit == 0) {
    throw std::runtime_error("Embedding context or batch size is zero");
  }

  if (tokens.size() > limit) {
    Logger::Warning("嵌入输入超过长度限制，token 数=", tokens.size(),
                    "，截断至 ", limit);
    tokens.resize(limit);
  }

  llama_memory_clear(llama_get_memory(ctx), true);

  llama_batch batch =
      llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);

  batch.n_tokens = static_cast<int32_t>(tokens.size());

  for (std::size_t i = 0; i < tokens.size(); ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);

    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;

    batch.logits[i] = (i + 1 == tokens.size()) ? 1 : 0;
  }

  int decodeResult = llama_decode(ctx, batch);

  if (decodeResult != 0) {
    llama_batch_free(batch);
    throw std::runtime_error("llama_decode failed while generating embedding");
  }

  const float* embeddingData = llama_get_embeddings_seq(ctx, 0);

  if (embeddingData == nullptr) {
    embeddingData = llama_get_embeddings(ctx);
  }

  if (embeddingData == nullptr) {
    llama_batch_free(batch);
    throw std::runtime_error("Failed to get embedding from llama context");
  }

  int embeddingSize = llama_model_n_embd(model);

  std::vector<float> embedding(embeddingData, embeddingData + embeddingSize);

  llama_batch_free(batch);

  return embedding;
}

void EmbeddingEngine::embedChunks(std::vector<Chunk>& chunks) {
  Logger::Info("开始生成文档向量，文本块数=", chunks.size());
  for (Chunk& chunk : chunks) {
    chunk.embedding = generateEmbedding(chunk.text);
  }
  Logger::Info("文档向量生成完成，文本块数=", chunks.size());
}

std::string EmbeddingEngine::sanitizeUtf8(const std::string& text) const {
  std::string result;

  // 先预留 text.size() 个字节的空间，result 不会超过这个大小
  result.reserve(text.size());

  std::size_t i = 0;

  while (i < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[i]);

    // 1 字节 ASCII，因为 ASCII码一共128个，因此小于 0x7f 的可以被认为是
    // ASCII码表对应的字符
    if (c <= 0x7F) {
      result.push_back(text[i]);
      ++i;
      continue;
    }

    // 2 字节 UTF-8，二字节 UTF-8 的标准是：110xxxxx 10xxxxxx
    if (c >= 0xC2 && c <= 0xDF) {
      if (i + 1 < text.size()) {
        unsigned char c1 = static_cast<unsigned char>(text[i + 1]);

        if (c1 >= 0x80 && c1 <= 0xBF) {
          result.append(text, i, 2);
          i += 2;
          continue;
        }
      }

      ++i;
      continue;
    }

    // 3 字节 UTF-8，三字节 UTF-8 的标准是：1110xxxx 10xxxxxx 10xxxxxx
    if (c >= 0xE0 && c <= 0xEF) {
      if (i + 2 < text.size()) {
        unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(text[i + 2]);

        bool valid = false;

        // 三字节开头的字节 0xE0 和 0xED 有特殊情况
        if (c == 0xE0) {
          valid = (c1 >= 0xA0 && c1 <= 0xBF) && (c2 >= 0x80 && c2 <= 0xBF);
        } else if (c == 0xED) {
          valid = (c1 >= 0x80 && c1 <= 0x9F) && (c2 >= 0x80 && c2 <= 0xBF);
        } else {
          valid = (c1 >= 0x80 && c1 <= 0xBF) && (c2 >= 0x80 && c2 <= 0xBF);
        }

        if (valid) {
          result.append(text, i, 3);
          i += 3;
          continue;
        }
      }

      ++i;
      continue;
    }

    // 4 字节 UTF-8，四字节 UTF-8 的标准是：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if (c >= 0xF0 && c <= 0xF4) {
      if (i + 3 < text.size()) {
        unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
        unsigned char c3 = static_cast<unsigned char>(text[i + 3]);

        bool valid = false;
        // 四字节开头的字节 0xF0 和 0xF4 有特殊情况
        if (c == 0xF0) {
          valid = (c1 >= 0x90 && c1 <= 0xBF) && (c2 >= 0x80 && c2 <= 0xBF) &&
                  (c3 >= 0x80 && c3 <= 0xBF);
        } else if (c == 0xF4) {
          valid = (c1 >= 0x80 && c1 <= 0x8F) && (c2 >= 0x80 && c2 <= 0xBF) &&
                  (c3 >= 0x80 && c3 <= 0xBF);
        } else {
          valid = (c1 >= 0x80 && c1 <= 0xBF) && (c2 >= 0x80 && c2 <= 0xBF) &&
                  (c3 >= 0x80 && c3 <= 0xBF);
        }

        if (valid) {
          result.append(text, i, 4);
          i += 4;
          continue;
        }
      }

      ++i;
      continue;
    }

    // 其他非法字节直接跳过
    ++i;
  }

  return result;
}
