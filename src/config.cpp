#include "config.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "logger.hpp"

namespace {

using nlohmann::json;

// 先检查整数类型和范围，避免浮点截断或负数转成无符号整数。
std::uint64_t ReadUnsigned(const json& value, const std::string& key,
                           std::uint64_t maximum) {
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
    throw std::invalid_argument(key + " 必须是非负整数。");
  }
  const auto number = value.get<std::uint64_t>();
  if (number > maximum) {
    throw std::invalid_argument(key + " 超出支持范围。");
  }
  return number;
}

std::string Trim(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

}  // namespace

AppConfig::AppConfig(const std::string& path) {
  std::ifstream input(std::filesystem::u8path(path));
  if (!input) {
    throw std::runtime_error("无法读取配置文件: " + path);
  }
  try {
    input >> values_;
    validate(values_);
    Logger::Info("配置读取完成: ", path);
  } catch (const std::exception& error) {
    throw std::runtime_error("配置文件无效: " + path + "，" + error.what());
  }
}

void AppConfig::validate(const json& values) {
  for (const char* section :
       {"models", "retrieval", "generation", "embedding"}) {
    if (!values.at(section).is_object()) {
      throw std::invalid_argument(std::string(section) + " 必须是对象。");
    }
  }
  const auto& models = values.at("models");
  for (const char* key : {"embedding", "generation"}) {
    if (Trim(models.at(key).get<std::string>()).empty()) {
      throw std::invalid_argument(std::string("models.") + key + " 不能为空。");
    }
  }
  ReadUnsigned(models.at("n_gpu_layers"), "models.n_gpu_layers",
               std::numeric_limits<int>::max());

  const auto positive_int = [](const json& value, const std::string& key) {
    if (ReadUnsigned(value, key, std::numeric_limits<int>::max()) == 0) {
      throw std::invalid_argument(key + " 必须大于 0。");
    }
  };
  positive_int(values.at("retrieval").at("top_k"), "retrieval.top_k");
  for (const char* section : {"generation", "embedding"}) {
    for (const char* key : {"n_ctx", "n_batch", "n_ubatch"}) {
      positive_int(values.at(section).at(key),
                   std::string(section) + "." + key);
    }
  }
  const auto& generation = values.at("generation");
  positive_int(generation.at("max_output_tokens"),
               "generation.max_output_tokens");
  positive_int(generation.at("top_k"), "generation.top_k");
  for (const char* key : {"temperature", "top_p"}) {
    if (!generation.at(key).is_number() ||
        !std::isfinite(generation.at(key).get<double>())) {
      throw std::invalid_argument(std::string("generation.") + key +
                                  " 必须是有限数字。");
    }
  }
  if (generation.at("temperature").get<double>() < 0.0) {
    throw std::invalid_argument("temperature 不能小于 0。");
  }
  const double top_p = generation.at("top_p").get<double>();
  if (top_p <= 0.0 || top_p > 1.0) {
    throw std::invalid_argument("top_p 必须在 (0, 1] 范围内。");
  }
}

AppConfig* AppConfig::update(const json& patch) {
  auto updated = values_;
  updated.merge_patch(patch);
  validate(updated);
  values_ = std::move(updated);
  for (const auto& section : patch.items()) {
    for (const auto& field : section.value().items()) {
      Logger::Info("配置项已更新: ", section.key(), ".", field.key());
    }
  }
  return this;
}

AppConfig* AppConfig::setEmbModelPath(const std::string& path) {
  return update({{"models", {{"embedding", path}}}});
}

AppConfig* AppConfig::setGenModelPath(const std::string& path) {
  return update({{"models", {{"generation", path}}}});
}

AppConfig* AppConfig::setGpuLayers(int n_gpu_layers) {
  return update({{"models", {{"n_gpu_layers", n_gpu_layers}}}});
}

AppConfig* AppConfig::setRetrievalTopK(int top_k) {
  return update({{"retrieval", {{"top_k", top_k}}}});
}

AppConfig* AppConfig::setGenNCtx(int n_ctx) {
  return update({{"generation", {{"n_ctx", n_ctx}}}});
}

AppConfig* AppConfig::setGenNBatch(int n_batch) {
  return update({{"generation", {{"n_batch", n_batch}}}});
}

AppConfig* AppConfig::setGenNUbatch(int n_ubatch) {
  return update({{"generation", {{"n_ubatch", n_ubatch}}}});
}

AppConfig* AppConfig::setMaxOutputTokens(int max_tokens) {
  return update({{"generation", {{"max_output_tokens", max_tokens}}}});
}

AppConfig* AppConfig::setTemperature(double temperature) {
  return update({{"generation", {{"temperature", temperature}}}});
}

AppConfig* AppConfig::setSamplerTopK(int top_k) {
  return update({{"generation", {{"top_k", top_k}}}});
}

AppConfig* AppConfig::setTopP(double top_p) {
  return update({{"generation", {{"top_p", top_p}}}});
}

AppConfig* AppConfig::setEmbNCtx(int n_ctx) {
  return update({{"embedding", {{"n_ctx", n_ctx}}}});
}

AppConfig* AppConfig::setEmbNBatch(int n_batch) {
  return update({{"embedding", {{"n_batch", n_batch}}}});
}

AppConfig* AppConfig::setEmbNUbatch(int n_ubatch) {
  return update({{"embedding", {{"n_ubatch", n_ubatch}}}});
}

AppConfig* AppConfig::adjustConfig(const std::string& key,
                                   const std::string& value) {
  static const std::map<std::string, std::string> aliases = {
      {"emb_model_path", "models.embedding"},
      {"gen_model_path", "models.generation"},
      {"n_gpu_layers", "models.n_gpu_layers"},
      {"top_k", "retrieval.top_k"},
      {"gen_n_ctx", "generation.n_ctx"},
      {"gen_n_batch", "generation.n_batch"},
      {"gen_n_ubatch", "generation.n_ubatch"},
      {"max_output_tokens", "generation.max_output_tokens"},
      {"temperature", "generation.temperature"},
      {"top_k_sampler", "generation.top_k"},
      {"top_p", "generation.top_p"},
      {"emb_n_ctx", "embedding.n_ctx"},
      {"emb_n_batch", "embedding.n_batch"},
      {"emb_n_ubatch", "embedding.n_ubatch"}};
  std::string name = Trim(key);
  const auto alias = aliases.find(name);
  if (alias != aliases.end()) {
    name = alias->second;
  }
  const auto separator = name.find('.');
  if (separator == std::string::npos) {
    throw std::invalid_argument("不支持的配置项: " + name);
  }
  const auto section = name.substr(0, separator);
  const auto field = name.substr(separator + 1);
  if (!values_.contains(section) || !values_.at(section).is_object() ||
      !values_.at(section).contains(field)) {
    throw std::invalid_argument("不支持的配置项: " + name);
  }
  json parsed;
  const auto& old_value = values_.at(section).at(field);
  const std::string raw = Trim(value);
  if (old_value.is_string()) {
    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
      parsed = json::parse(raw);
    } else if (raw.size() >= 2 && raw.front() == '\'' && raw.back() == '\'') {
      parsed = raw.substr(1, raw.size() - 2);
    } else {
      parsed = raw;
    }
    if (!parsed.is_string()) {
      throw std::invalid_argument(name + " 必须是字符串。");
    }
  } else {
    parsed = json::parse(raw);
    const bool floating =
        name == "generation.temperature" || name == "generation.top_p";
    if ((floating && !parsed.is_number()) ||
        (!floating && old_value.is_number_integer() &&
         !parsed.is_number_integer()) ||
        (old_value.is_number() && !parsed.is_number())) {
      throw std::invalid_argument(name + " 的数值类型不正确。");
    }
  }
  return update({{section, {{field, parsed}}}});
}

bool AppConfig::saveToFile(const std::string& path) const {
  std::ofstream output(std::filesystem::u8path(path));
  if (!output) {
    Logger::Error("[Config] 无法写入配置文件 ", path, '\n');
    return false;
  }
  output << values_.dump(4) << '\n';
  output.flush();
  if (!output) {
    Logger::Error("写入配置文件失败: ", path);
    return false;
  }
  Logger::Info("配置已保存: ", path);
  return true;
}

void AppConfig::printConfig(std::ostream& out) const {
  out << "当前 AppConfig 对象配置如下：\n" << values_.dump(4) << '\n';
}

nlohmann::json AppConfig::ToJson() const { return values_; }

const std::string& AppConfig::embModelPath() const {
  return values_.at("models").at("embedding").get_ref<const std::string&>();
}
const std::string& AppConfig::genModelPath() const {
  return values_.at("models").at("generation").get_ref<const std::string&>();
}
int AppConfig::nGpuLayers() const {
  return values_.at("models").at("n_gpu_layers").get<int>();
}
int AppConfig::retrievalTopK() const {
  return values_.at("retrieval").at("top_k").get<int>();
}
int AppConfig::genNCtx() const {
  return values_.at("generation").at("n_ctx").get<int>();
}
int AppConfig::genNBatch() const {
  return values_.at("generation").at("n_batch").get<int>();
}
int AppConfig::genNUbatch() const {
  return values_.at("generation").at("n_ubatch").get<int>();
}
int AppConfig::maxOutputTokens() const {
  return values_.at("generation").at("max_output_tokens").get<int>();
}
double AppConfig::getTemperature() const {
  return values_.at("generation").at("temperature").get<double>();
}
int AppConfig::samplerTopK() const {
  return values_.at("generation").at("top_k").get<int>();
}
double AppConfig::topP() const {
  return values_.at("generation").at("top_p").get<double>();
}
int AppConfig::embNCtx() const {
  return values_.at("embedding").at("n_ctx").get<int>();
}
int AppConfig::embNBatch() const {
  return values_.at("embedding").at("n_batch").get<int>();
}
int AppConfig::embNUbatch() const {
  return values_.at("embedding").at("n_ubatch").get<int>();
}
