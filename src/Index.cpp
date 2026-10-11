#include "Index.hpp"
#include "Text.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
namespace rag {
void Index::prepare() {
    postings_.clear();
    lengths_.clear();
    std::set<std::string> ids;
    std::size_t dim = chunks.empty() ? 0 : chunks.front().embedding.size();
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        auto &c = chunks[i];
        if (!ids.insert(c.id).second)
            throw std::runtime_error("Duplicate chunk ID");
        if (c.embedding.size() != dim)
            throw std::runtime_error("Mixed embedding dimensions in index");
        for (float x : c.embedding)
            if (!std::isfinite(x))
                throw std::runtime_error("Non-finite embedding");
        auto tokens = terms(c.title + "\n" + c.text);
        lengths_.push_back(tokens.size());
        std::unordered_map<std::string, int> counts;
        for (const auto &t : tokens)
            ++counts[t];
        for (const auto &kv : counts)
            postings_[kv.first].push_back({i, kv.second});
    }
    if (chunks.empty())
        throw std::runtime_error("Empty index");
    if ((dim == 0) != embedding_model.empty())
        throw std::runtime_error("Embedding metadata does not match vectors");
    average_ = std::max(
        1.0, static_cast<double>(std::accumulate(lengths_.begin(), lengths_.end(), std::size_t{})) /
                 chunks.size());
}
void Index::save(const std::string &path) const {
    nlohmann::json data = {{"format", "rag-index"},
                           {"version", 1},
                           {"tokenizer", "utf8-bigram-v1"},
                           {"embedding_model", embedding_model},
                           {"chunks", chunks}};
    data["checksum"] = fingerprint(data["chunks"].dump());
    writeJson(std::filesystem::u8path(path), data);
}
Index Index::load(const std::string &path) {
    auto data = nlohmann::json::parse(readText(std::filesystem::u8path(path)));
    if (data.at("format") != "rag-index" || data.at("version") != 1 ||
        data.at("tokenizer") != "utf8-bigram-v1")
        throw std::runtime_error("Unsupported index format/version/tokenizer");
    if (data.at("checksum") != fingerprint(data.at("chunks").dump()))
        throw std::runtime_error("Index checksum mismatch");
    Index out;
    data.at("chunks").get_to(out.chunks);
    data.at("embedding_model").get_to(out.embedding_model);
    out.prepare();
    return out;
}
std::vector<std::pair<std::size_t, double>> Index::lexical(const std::string &query,
                                                           const RetrievalConfig &config) const {
    std::vector<double> scores(chunks.size(), 0);
    std::set<std::string> unique;
    for (const auto &t : terms(query))
        unique.insert(t);
    for (const auto &t : unique) {
        auto it = postings_.find(t);
        if (it == postings_.end())
            continue;
        double df = static_cast<double>(it->second.size());
        double idf = std::log(1 + (chunks.size() - df + 0.5) / (df + 0.5));
        double weight = static_cast<unsigned char>(t[0]) >= 128
                            ? (characters(t).size() == 1 ? 0.15 : 1.0)
                            : 1.2;
        for (const auto &item : it->second) {
            double tf = item.second,
                   norm = config.bm25_k1 *
                          (1 - config.bm25_b + config.bm25_b * lengths_[item.first] / average_);
            scores[item.first] += weight * idf * tf * (config.bm25_k1 + 1) / (tf + norm);
        }
    }
    std::vector<std::pair<std::size_t, double>> result;
    for (std::size_t i = 0; i < scores.size(); ++i)
        if (scores[i] > 0)
            result.push_back({i, scores[i]});
    std::sort(result.begin(), result.end(), [](auto a, auto b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (result.size() > static_cast<std::size_t>(config.candidate_k))
        result.resize(config.candidate_k);
    return result;
}
std::vector<Hit> Index::search(const std::string &query, const std::vector<float> &vector,
                               const RetrievalConfig &config) const {
    std::vector<double> fused(chunks.size(), 0), bm(chunks.size(), 0), cos(chunks.size(), 0);
    auto add = [&](const std::vector<std::pair<std::size_t, double>> &ranked, double weight,
                   bool islex) {
        for (std::size_t r = 0; r < ranked.size(); ++r) {
            auto i = ranked[r].first;
            fused[i] += weight / (config.rrf_k + r + 1);
            if (islex)
                bm[i] = std::max(bm[i], ranked[r].second);
            else
                cos[i] = ranked[r].second;
        }
    };
    add(lexical(query, config), 1, true);
    auto stem = questionStem(query);
    if (stem != query)
        add(lexical(stem, config), 1, true);
    if (config.option_queries)
        for (const auto &option : parseOptions(query))
            add(lexical(stem + "\n" + option.text, config), 0.25, true);
    if (!vector.empty()) {
        if (chunks.front().embedding.size() != vector.size())
            throw std::runtime_error("Query embedding dimension mismatch");
        double qnorm = 0;
        for (float x : vector) {
            if (!std::isfinite(x))
                throw std::runtime_error("Non-finite query embedding");
            qnorm += x * x;
        }
        if (qnorm <= 0)
            throw std::runtime_error("Zero query embedding");
        std::vector<std::pair<std::size_t, double>> ranked;
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            double dot = 0, norm = 0;
            for (std::size_t j = 0; j < vector.size(); ++j) {
                dot += vector[j] * chunks[i].embedding[j];
                norm += chunks[i].embedding[j] * chunks[i].embedding[j];
            }
            double score = norm > 0 ? dot / std::sqrt(norm * qnorm) : 0;
            ranked.push_back({i, score});
        }
        std::sort(ranked.begin(), ranked.end(), [](auto a, auto b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        if (ranked.size() > static_cast<std::size_t>(config.candidate_k))
            ranked.resize(config.candidate_k);
        add(ranked, config.vector_weight, false);
    }
    std::vector<Hit> hits;
    for (std::size_t i = 0; i < fused.size(); ++i)
        if (fused[i] > 0)
            hits.push_back({i, fused[i], bm[i], cos[i]});
    std::sort(hits.begin(), hits.end(), [](const Hit &a, const Hit &b) {
        return a.score != b.score ? a.score > b.score : a.index < b.index;
    });
    if (hits.size() > static_cast<std::size_t>(config.top_k))
        hits.resize(config.top_k);
    return hits;
}
nlohmann::json Index::evidence(const std::vector<Hit> &hits) const {
    auto data = nlohmann::json::array();
    for (const auto &h : hits) {
        const auto &c = chunks.at(h.index);
        data.push_back({{"chunk_id", c.id},
                        {"record_id", c.record_id},
                        {"source", c.source},
                        {"title", c.title},
                        {"text", c.text},
                        {"rrf_score", h.score},
                        {"bm25_score", h.lexical},
                        {"cosine", h.cosine}});
    }
    return data;
}
} // namespace rag
