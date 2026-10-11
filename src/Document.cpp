#include "Document.hpp"
#include "Logger.hpp"
#include "Text.hpp"
#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
namespace rag {
void to_json(nlohmann::json &j, const Chunk &c) {
    j = {{"id", c.id},       {"source", c.source}, {"record_id", c.record_id},
         {"title", c.title}, {"text", c.text},     {"embedding", c.embedding}};
}
void from_json(const nlohmann::json &j, Chunk &c) {
    j.at("id").get_to(c.id);
    j.at("source").get_to(c.source);
    j.at("record_id").get_to(c.record_id);
    j.at("title").get_to(c.title);
    j.at("text").get_to(c.text);
    j.at("embedding").get_to(c.embedding);
}
DocumentProcessor::DocumentProcessor(DocumentConfig config) : config_(config) {
}
void DocumentProcessor::split(const std::string &text, const std::string &source,
                              const std::string &record, const std::string &title,
                              std::vector<Chunk> &out) const {
    auto cs = characters(text);
    std::size_t start = 0, part = 0;
    while (start < cs.size()) {
        auto end = std::min(cs.size(), start + static_cast<std::size_t>(config_.chunk_size));
        if (end < cs.size())
            for (auto p = end; p > start + static_cast<std::size_t>(config_.chunk_size / 2); --p)
                if (cs[p - 1] == "。" || cs[p - 1] == "\n" || cs[p - 1] == "；") {
                    end = p;
                    break;
                }
        std::string content;
        for (auto i = start; i < end; ++i)
            content += cs[i];
        if (content.find_first_not_of(" \t\n") != std::string::npos) {
            Chunk c;
            c.source = source;
            c.record_id = record;
            c.title = title;
            c.text = content;
            c.id = fingerprint(source + "\n" + record + "\n" + std::to_string(part++));
            out.push_back(std::move(c));
        }
        if (end == cs.size())
            break;
        start = std::max(start + 1, end > static_cast<std::size_t>(config_.overlap_size)
                                        ? end - static_cast<std::size_t>(config_.overlap_size)
                                        : end);
    }
}
void DocumentProcessor::loadFile(const std::filesystem::path &path, std::vector<Chunk> &out) const {
    auto source = std::filesystem::weakly_canonical(path).u8string();
    auto text = readText(path);
    auto ext = path.extension().u8string();
    if (ext == ".json" || (ext.empty() && text.find_first_not_of(" \n\t") != std::string::npos &&
                           text[text.find_first_not_of(" \n\t")] == '[')) {
        auto data = nlohmann::json::parse(text);
        if (data.is_object() && data.contains("documents"))
            data = data.at("documents");
        if (data.is_object())
            data = nlohmann::json::array({data});
        if (!data.is_array())
            throw std::runtime_error("Document JSON must be an object, array or {documents:[...]}");
        std::set<std::string> ids;
        for (std::size_t i = 0; i < data.size(); ++i) {
            const auto &row = data[i];
            if (!row.is_object())
                throw std::runtime_error("Document record must be an object");
            if (row.contains("question_text") || row.contains("answer") ||
                row.contains("refered_rules"))
                throw std::runtime_error("Question/answer records cannot be ingested: " + source);
            std::string body, key;
            for (const auto &k : {"text", "content", "rule_text"})
                if (row.contains(k)) {
                    body = row.at(k).get<std::string>();
                    break;
                }
            for (const auto &k : {"id", "doc_id", "rule_id"})
                if (row.contains(k)) {
                    key = row[k].is_string() ? row[k].get<std::string>() : row[k].dump();
                    break;
                }
            if (key.empty())
                key = std::to_string(i + 1);
            if (!ids.insert(key).second)
                throw std::runtime_error("Duplicate record ID: " + key + " in " + source);
            if (body.find_first_not_of(" \n\t") == std::string::npos) {
                Logger::warning("Skipping empty record: " + key);
                continue;
            }
            auto title = row.value("title", std::string{});
            // JSON records are atomic evidence units. Keep their boundary, never merge adjacent
            // records.
            Chunk c;
            c.source = source;
            c.record_id = key;
            c.title = title;
            c.text = body;
            c.id = fingerprint(source + "\n" + key);
            characters(body);
            out.push_back(std::move(c));
        }
    } else {
        std::istringstream stream(text);
        std::string line, section, title;
        std::size_t n = 0;
        auto flush = [&]() {
            if (!section.empty()) {
                split(section, source, std::to_string(++n), title, out);
                section.clear();
            }
        };
        while (std::getline(stream, line)) {
            bool heading = (!line.empty() && line[0] == '#') ||
                           (line.rfind("第", 0) == 0 && (line.find("章") != std::string::npos ||
                                                         line.find("回") != std::string::npos)) ||
                           line.rfind("Chapter ", 0) == 0;
            if (heading) {
                flush();
                title = line;
            }
            section += line + "\n";
        }
        flush();
    }
}
std::vector<Chunk> DocumentProcessor::load(const std::vector<std::string> &paths) const {
    std::set<std::filesystem::path> files;
    for (const auto &name : paths) {
        auto p = std::filesystem::u8path(name);
        if (std::filesystem::is_directory(p)) {
            for (const auto &e : std::filesystem::recursive_directory_iterator(p))
                if (e.is_regular_file()) {
                    auto ext = e.path().extension().u8string();
                    if (ext == ".txt" || ext == ".md" || ext == ".json")
                        files.insert(std::filesystem::weakly_canonical(e.path()));
                }
        } else if (std::filesystem::is_regular_file(p))
            files.insert(std::filesystem::weakly_canonical(p));
        else
            throw std::runtime_error("Input does not exist: " + name);
    }
    std::vector<Chunk> result;
    for (const auto &p : files)
        loadFile(p, result);
    if (result.empty())
        throw std::runtime_error("No non-empty documents found");
    return result;
}
} // namespace rag
