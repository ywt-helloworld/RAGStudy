#pragma once
#include "Index.hpp"
#include <memory>
#include <string>
namespace rag {
class Inference {
  public:
    explicit Inference(const Config &config);
    ~Inference();
    Inference(const Inference &) = delete;
    Inference &operator=(const Inference &) = delete;
    std::vector<float> embed(const std::string &text);
    std::string answer(const std::string &query, const Index &index, const std::vector<Hit> &hits,
                       bool choice, nlohmann::json &trace);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace rag
