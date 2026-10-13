#pragma once

#include <llama.h>

#include <string>
#include <vector>

#include "config.hpp"

enum class ModelMode { Embedding, Generation };

class LlamaModelBase {
 public:
  LlamaModelBase(const AppConfig& config, ModelMode mode);
  virtual ~LlamaModelBase();
  // 禁用拷贝和移动
  LlamaModelBase(const LlamaModelBase& other) = delete;
  LlamaModelBase& operator=(const LlamaModelBase& other) = delete;
  LlamaModelBase(LlamaModelBase&& other) = delete;
  LlamaModelBase& operator=(LlamaModelBase&& other) = delete;

 protected:
  // 将文本转换为 llama token，供嵌入和生成模块复用
  std::vector<llama_token> tokenize(const std::string& text, bool addSpecial,
                                    bool parseSpecial) const;

  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  const llama_vocab* vocab = nullptr;

  int nCtx = 0;
  int nBatch = 0;
  int nUbatch = 0;

  ModelMode mode;
};