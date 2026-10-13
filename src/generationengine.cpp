#include "generationengine.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "logger.hpp"

namespace {  // 用于针对不同的大模型生成不同种的 prompt

std::string toLowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  return s;
}

bool containsText(const std::string& text, const std::string& pattern) {
  return text.find(pattern) != std::string::npos;
}

// 根据不同的大模型生成不同的提示词
std::string buildPromptByModel(const std::string& modelPath,
                               const std::string& context,
                               const std::string& query,
                               bool single_choice = false) {
  const std::string lowerPath = toLowerCopy(modelPath);

  std::string systemPrompt =
      "你是一个严谨的中文知识问答助手。"
      "你必须只使用中文回答。"
      "你必须根据用户提供的[参考上下文]回答问题。"
      "不要凭空编造，不要使用参考上下文之外的信息。"
      "如果参考上下文中没有足够信息回答问题，请明确说明：根据当前片段无法得知。"
      "你的回答必须包含两部分：【分析依据】和【最终答案】。"
      "【分析依据】中要简要说明你参考了哪些片段，这些片段提供了什么线索，以及这"
      "些线索如何支持你的判断。"
      "【最终答案】中要直接回答用户问题，语言简洁明确。"
      "不要输出英文，不要输出特殊标记。";

  if (single_choice) {
    systemPrompt =
        "你是一个严谨的中文知识问答助手。仅根据参考上下文回答单选题。"
        "可以先用中文简要说明【分析依据】，然后输出【最终答案】。"
        "【最终答案】后只能有一个 A、B、C 或 D 字母，不要重复选项正文。"
        "如果无法判断，请明确说明无法判断。";
  }

  const std::string userPrompt =
      "下面是系统从知识库中检索到的参考上下文：\n\n" + context +
      "\n\n用户问题：" + query +
      "\n\n请严格按照下面格式回答：\n"
      "【分析依据】\n"
      "1. 说明哪些片段与问题有关。\n"
      "2. 说明这些片段中出现了哪些关键信息。\n"
      "3. 说明如何根据这些信息得到答案。\n\n"
      "【最终答案】\n" +
      (single_choice ? "仅输出一个 A/B/C/D 选项字母，不要添加解释或其他结论。"
                     : "直接给出答案。");

  // Qwen / ChatML 类模型
  if (containsText(lowerPath, "qwen") || containsText(lowerPath, "chatml")) {
    return "<|im_start|>system\n" + systemPrompt +
           "<|im_end|>\n"
           "<|im_start|>user\n" +
           userPrompt +
           "<|im_end|>\n"
           "<|im_start|>assistant\n";
  }

  // OpenAI gpt-oss / Harmony 类模型
  if (containsText(lowerPath, "openai") || containsText(lowerPath, "gpt-oss") ||
      containsText(lowerPath, "gpt_oss")) {
    return "<|start|>system<|message|>"
           "你是一个严谨的中文知识问答助手。"
           "你必须只使用中文回答。"
           "你必须根据参考上下文回答问题。"
           "回答必须包含【分析依据】和【最终答案】两部分。"
           "不要输出英文，不要输出特殊标记。"
           "不要输出 <|start|>、<|end|>、<|channel|>、<|message|>。"
           "<|end|>"
           "<|start|>developer<|message|># Instructions\n" +
           systemPrompt +
           "\n请在 final 回复中用中文输出简短的【分析依据】和【最终答案】。"
           "\n【分析依据】只说明可展示给用户的依据，不要输出底层推理通道。"
           "\n不要输出 analysis 标签，不要输出 harmony 特殊标记。"
           "<|end|>"
           "<|start|>user<|message|>" +
           userPrompt +
           "<|end|>"
           "<|start|>assistant<|channel|>final<|message|>";
  }

  // Llama 3 Instruct 类模型
  if (containsText(lowerPath, "llama-3") || containsText(lowerPath, "llama3")) {
    return "<|begin_of_text|>"
           "<|start_header_id|>system<|end_header_id|>\n\n" +
           systemPrompt +
           "<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n\n" +
           userPrompt +
           "<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n\n";
  }

  // Alpaca / Vicuna 风格模型
  if (containsText(lowerPath, "alpaca") || containsText(lowerPath, "vicuna")) {
    return "### Instruction:\n" + systemPrompt + "\n\n### Input:\n" +
           userPrompt + "\n\n### Response:\n";
  }

  // 默认兜底模板：不依赖特殊 token，兼容性最好
  return "系统说明：\n" + systemPrompt + "\n\n用户问题与参考上下文：\n" +
         userPrompt + "\n\n助手回答：\n";
}

}  // namespace

// 把模型生成出来的 token 编号，转换成可以打印出来的文字片段
static std::string tokenToPiece(const llama_vocab* vocab, llama_token token) {
  if (vocab == nullptr) {
    return {};
  }

  std::vector<char> buffer(128);

  int n = llama_token_to_piece(vocab, token, buffer.data(),
                               static_cast<int32_t>(buffer.size()), 0, true);

  if (n < 0) {
    buffer.resize(static_cast<std::size_t>(-n));
    n = llama_token_to_piece(vocab, token, buffer.data(),
                             static_cast<int32_t>(buffer.size()), 0, true);
  }

  if (n <= 0) {
    return {};
  }

  return std::string(buffer.data(), static_cast<std::size_t>(n));
}

GenerationEngine::GenerationEngine(const AppConfig& config)
    : LlamaModelBase(config, ModelMode::Generation),
      maxOutputTokens(config.maxOutputTokens()),
      temperature(config.getTemperature()),
      samplerTopK(config.samplerTopK()),
      topP(config.topP()),
      genModelPath(config.genModelPath()) {}

std::string GenerationEngine::buildContext(
    const std::vector<SearchResult>& results) const {
  // 使用 ostringstream 便于拼接字符串
  std::ostringstream oss;

  if (results.empty()) {
    oss << "当前没有检索到相关片段。\n";
    return oss.str();
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    const SearchResult& result = results[i];

    if (result.chunk == nullptr) {
      continue;
    }

    oss << "[片段 " << (i + 1) << "]\n";
    oss << "出处：" << result.chunk->metadata << "\n";
    oss << "相似度：" << std::fixed << std::setprecision(4) << result.score
        << "\n";
    oss << "内容：" << result.chunk->text << "\n\n";
  }

  return oss.str();
}

void GenerationEngine::generateStream(
    const std::string& query, const std::vector<SearchResult>& results) {
  Generate(query, results, true);
}

GenerationResult GenerationEngine::Generate(
    const std::string& query, const std::vector<SearchResult>& results,
    bool stream, bool single_choice) {
  Logger::Info("开始生成回答，证据块数=", results.size());
  GenerationResult result;
  const auto fail = [&](const std::string& error) {
    result.error = error;
    Logger::Error(error);
    return result;
  };
  const std::string context = buildContext(results);
  if (stream) {
    std::cout << "\n========== 检索到的 Top " << results.size()
              << " 个 Chunk ==========\n"
              << context << std::endl;
  }
  result.prompt =
      buildPromptByModel(genModelPath, context, query, single_choice);
  const auto prompt_tokens = tokenize(result.prompt, true, true);
  const auto limit = static_cast<std::size_t>(std::min(nCtx, nBatch));
  if (prompt_tokens.empty() || limit == 0) {
    return fail("生成提示为空或上下文/批次大小为零。");
  }
  if (prompt_tokens.size() > limit) {
    return fail(
        "生成提示超长，token 数=" + std::to_string(prompt_tokens.size()) +
        "，上限=" + std::to_string(limit));
  }

  llama_memory_clear(llama_get_memory(ctx), true);
  struct BatchGuard {
    llama_batch value;
    ~BatchGuard() { llama_batch_free(value); }
  } batch_guard{
      llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1)};
  auto& batch = batch_guard.value;
  batch.n_tokens = static_cast<int32_t>(prompt_tokens.size());
  for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
    batch.token[i] = prompt_tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i + 1 == prompt_tokens.size()) ? 1 : 0;
  }
  int decode_result = llama_decode(ctx, batch);
  if (decode_result != 0) {
    return fail("生成提示预填充失败，code=" + std::to_string(decode_result));
  }
  auto sampler_params = llama_sampler_chain_default_params();
  sampler_params.no_perf = false;
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
      llama_sampler_chain_init(sampler_params), llama_sampler_free);
  if (!sampler) {
    return fail("无法创建生成采样器。");
  }
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(samplerTopK));
  llama_sampler_chain_add(
      sampler.get(), llama_sampler_init_top_p(static_cast<float>(topP), 1));
  llama_sampler_chain_add(
      sampler.get(), llama_sampler_init_temp(static_cast<float>(temperature)));
  llama_sampler_chain_add(sampler.get(),
                          llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  auto current_pos = static_cast<std::int32_t>(prompt_tokens.size());
  if (stream) {
    std::cout << "AI 回答> ";
  }
  int generated_tokens = 0;
  for (int i = 0; i < maxOutputTokens; ++i) {
    const auto token = llama_sampler_sample(sampler.get(), ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
      break;
    }
    ++generated_tokens;
    const auto piece = tokenToPiece(vocab, token);
    result.text += piece;
    if (stream) {
      std::cout << piece << std::flush;
    }
    if (i + 1 == maxOutputTokens) {
      break;
    }
    if (current_pos >= nCtx) {
      return fail("生成回答超过上下文长度限制。");
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = static_cast<llama_pos>(current_pos);
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
      return fail("回答生成失败，code=" + std::to_string(decode_result));
    }
    ++current_pos;
  }
  if (stream) {
    std::cout << std::endl;
  }
  Logger::Info("回答生成完成，输出 token 数=", generated_tokens);
  return result;
}
