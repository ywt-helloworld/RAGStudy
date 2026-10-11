#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace rag {
std::vector<std::string> characters(const std::string &text);
std::vector<std::string> terms(const std::string &text);
std::string clip(const std::string &text, std::size_t count);
std::string readText(const std::filesystem::path &path);
void writeJson(const std::filesystem::path &path, const nlohmann::json &data);
std::string fingerprint(const std::string &text);
std::string modelIdentity(const std::string &path);
struct Option {
    std::string label, text;
};
std::vector<Option> parseOptions(const std::string &question);
std::string questionStem(const std::string &question);
} // namespace rag
