#pragma once
#include "Document.hpp"
#include <string>
#include <unordered_map>
#include <vector>
namespace rag {
struct Hit {
    std::size_t index;
    double score, lexical, cosine;
};
class Index {
  public:
    std::vector<Chunk> chunks;
    std::string embedding_model;
    void prepare();
    void save(const std::string &path) const;
    static Index load(const std::string &path);
    std::vector<Hit> search(const std::string &query, const std::vector<float> &vector,
                            const RetrievalConfig &config) const;
    nlohmann::json evidence(const std::vector<Hit> &hits) const;

  private:
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, int>>> postings_;
    std::vector<std::size_t> lengths_;
    double average_ = 1;
    std::vector<std::pair<std::size_t, double>> lexical(const std::string &query,
                                                        const RetrievalConfig &config) const;
};
} // namespace rag
