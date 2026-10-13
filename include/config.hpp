#pragma once

#include <iosfwd>
#include <nlohmann/json.hpp>
#include <string>

class AppConfig {
 public:
  // 必须从配置文件构造；缺失字段、解析错误或无效参数会抛出异常。
  explicit AppConfig(const std::string& path);

  // 修改内存配置并返回 this，以便连续使用 -> 设置。
  AppConfig* setEmbModelPath(const std::string& path);
  AppConfig* setGenModelPath(const std::string& path);
  AppConfig* setGpuLayers(int n_gpu_layers);
  AppConfig* setRetrievalTopK(int top_k);
  AppConfig* setGenNCtx(int n_ctx);
  AppConfig* setGenNBatch(int n_batch);
  AppConfig* setGenNUbatch(int n_ubatch);
  AppConfig* setMaxOutputTokens(int max_tokens);
  AppConfig* setTemperature(double temperature);
  AppConfig* setSamplerTopK(int top_k);
  AppConfig* setTopP(double top_p);
  AppConfig* setEmbNCtx(int n_ctx);
  AppConfig* setEmbNBatch(int n_batch);
  AppConfig* setEmbNUbatch(int n_ubatch);
  AppConfig* adjustConfig(const std::string& key, const std::string& value);

  // 保存与修改分开；调用设置接口不会自动写文件。
  bool saveToFile(const std::string& path) const;
  void printConfig(std::ostream& out) const;
  nlohmann::json ToJson() const;

  const std::string& embModelPath() const;
  const std::string& genModelPath() const;
  int nGpuLayers() const;
  int retrievalTopK() const;
  int genNCtx() const;
  int genNBatch() const;
  int genNUbatch() const;
  int maxOutputTokens() const;
  double getTemperature() const;
  int samplerTopK() const;
  double topP() const;
  int embNCtx() const;
  int embNBatch() const;
  int embNUbatch() const;

 private:
  // 保留文件中的配置，不再维护第二套硬编码默认值。
  nlohmann::json values_;

  AppConfig* update(const nlohmann::json& patch);
  static void validate(const nlohmann::json& values);
};
