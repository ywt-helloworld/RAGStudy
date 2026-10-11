#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace rag {
struct Question {
    std::string id, text;
};
std::vector<Question> loadQuestions(const std::string &path);
nlohmann::json evaluate(const std::string &questions, const nlohmann::json &results);
} // namespace rag
