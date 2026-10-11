#pragma once
#include "Config.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace rag {
struct Chunk {
    std::string id, source, record_id, title, text;
    std::vector<float> embedding;
};
void to_json(nlohmann::json &j, const Chunk &c);
void from_json(const nlohmann::json &j, Chunk &c);
class DocumentProcessor {
  public:
    explicit DocumentProcessor(DocumentConfig config);
    std::vector<Chunk> load(const std::vector<std::string> &paths) const;

  private:
    DocumentConfig config_;
    void loadFile(const std::filesystem::path &path, std::vector<Chunk> &out) const;
    void split(const std::string &text, const std::string &source, const std::string &record,
               const std::string &title, std::vector<Chunk> &out) const;
};
} // namespace rag
