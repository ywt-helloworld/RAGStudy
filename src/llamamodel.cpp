#include "llamamodel.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "logger.hpp"

LlamaModelBase::LlamaModelBase(const AppConfig& config, ModelMode mode)
    : mode(mode) {
  std::string modelPath;

  // 根据模式从 config 里获取参数
  if (mode == ModelMode::Embedding) {
    modelPath = config.embModelPath();
    nCtx = config.embNCtx();
    nBatch = config.embNBatch();
    nUbatch = config.embNUbatch();
  } else {
    modelPath = config.genModelPath();
    nCtx = config.genNCtx();
    nBatch = config.genNBatch();
    nUbatch = config.genNUbatch();
  }

  // 根据 ModelMode （ Embedding / Generation
  // ）加载对应模型，创建上下⽂，同时在失败时做好资源清理
  llama_model_params modelParams = llama_model_default_params();
  modelParams.n_gpu_layers = config.nGpuLayers();

  Logger::Info("开始加载", mode == ModelMode::Embedding ? "嵌入" : "生成",
               "模型: ", modelPath, "，n_ctx=", nCtx,
               "，n_gpu_layers=", config.nGpuLayers());
  model = llama_model_load_from_file(modelPath.c_str(), modelParams);

  if (model == nullptr) {
    throw std::runtime_error("Failed to load model: " + modelPath);
  }

  vocab = llama_model_get_vocab(model);

  if (vocab == nullptr) {
    llama_model_free(model);
    model = nullptr;

    throw std::runtime_error("Failed to get vocabulary from model: " +
                             modelPath);
  }

  llama_context_params ctxParams = llama_context_default_params();

  ctxParams.n_ctx = static_cast<std::uint32_t>(nCtx);
  ctxParams.n_batch = static_cast<std::uint32_t>(nBatch);
  ctxParams.n_ubatch = static_cast<std::uint32_t>(nUbatch);
  ctxParams.n_seq_max = 1;

  if (mode == ModelMode::Embedding) {
    ctxParams.embeddings = true;
    ctxParams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  } else {
    ctxParams.embeddings = false;
  }

  ctx = llama_init_from_model(model, ctxParams);

  if (ctx == nullptr) {
    llama_model_free(model);
    model = nullptr;
    vocab = nullptr;

    throw std::runtime_error("Failed to create context for model: " +
                             modelPath);
  }
  Logger::Info("模型加载完成: ", modelPath);
}

LlamaModelBase::~LlamaModelBase() {
  if (ctx != nullptr) {
    llama_free(ctx);
    ctx = nullptr;
  }

  if (model != nullptr) {
    llama_model_free(model);
    model = nullptr;
  }

  vocab = nullptr;
}

std::vector<llama_token> LlamaModelBase::tokenize(const std::string& text,
                                                  bool addSpecial,
                                                  bool parseSpecial) const {
  if (vocab == nullptr) {
    throw std::runtime_error("Cannot tokenize: vocab is null");
  }

  if (text.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::runtime_error("Cannot tokenize: text is too long");
  }

  const std::int32_t textLength = static_cast<std::int32_t>(text.size());

  int tokenCount = llama_tokenize(vocab, text.c_str(), textLength, nullptr, 0,
                                  addSpecial, parseSpecial);

  if (tokenCount < 0) {
    tokenCount = -tokenCount;
  }

  if (tokenCount == 0) {
    return {};
  }

  std::vector<llama_token> tokens(static_cast<std::size_t>(tokenCount));

  int actualCount =
      llama_tokenize(vocab, text.c_str(), textLength, tokens.data(), tokenCount,
                     addSpecial, parseSpecial);

  if (actualCount < 0) {
    throw std::runtime_error("Failed to tokenize input text");
  }

  tokens.resize(static_cast<std::size_t>(actualCount));

  return tokens;
}
