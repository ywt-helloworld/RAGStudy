#include "Evaluation.hpp"
#include "Text.hpp"
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>
namespace rag {
std::vector<Question> loadQuestions(const std::string &path) {
    auto data = nlohmann::json::parse(readText(std::filesystem::u8path(path)));
    if (!data.is_array())
        throw std::runtime_error("Questions must be a JSON array");
    std::vector<Question> result;
    std::set<std::string> ids;
    for (const auto &row : data) {
        auto id = row.at("question_id").is_string() ? row.at("question_id").get<std::string>()
                                                    : row.at("question_id").dump();
        if (!ids.insert(id).second)
            throw std::runtime_error("Duplicate question_id: " + id);
        // Return only inference input. Gold answers and reference IDs cannot reach the pipeline.
        result.push_back({id, row.at("question_text").get<std::string>()});
    }
    return result;
}
nlohmann::json evaluate(const std::string &path, const nlohmann::json &results) {
    auto gold = nlohmann::json::parse(readText(std::filesystem::u8path(path)));
    if (!gold.is_array() || !results.is_array())
        throw std::runtime_error("Evaluation inputs must be arrays");
    std::unordered_map<std::string, nlohmann::json> byid;
    for (const auto &row : results) {
        auto id = row.at("question_id").get<std::string>();
        if (!byid.emplace(id, row).second)
            throw std::runtime_error("Duplicate result ID");
    }
    std::size_t covered = 0, labelled = 0, coveredlabelled = 0, correct = 0, invalid = 0,
                withrules = 0, allrules = 0, rulecount = 0, recalled = 0, single = 0, multi = 0,
                singleall = 0, multiall = 0;
    auto failures = nlohmann::json::array();
    for (const auto &row : gold) {
        auto id = row.at("question_id").is_string() ? row.at("question_id").get<std::string>()
                                                    : row.at("question_id").dump();
        auto it = byid.find(id);
        bool found = it != byid.end();
        if (found && it->second.contains("input_hash") &&
            it->second.at("input_hash") != fingerprint(row.at("question_text").get<std::string>()))
            throw std::runtime_error("Question text mismatch for ID " + id +
                                     ": results belong to another question set");
        if (found)
            ++covered;
        std::string prediction = found ? it->second.value("answer", std::string{}) : "";
        auto options = parseOptions(row.at("question_text").get<std::string>());
        bool valid = false;
        for (const auto &o : options)
            valid = valid || o.label == prediction;
        if (found && !options.empty() && !valid)
            ++invalid;
        bool answerok = true, rulesok = true;
        if (row.contains("answer")) {
            ++labelled;
            if (found)
                ++coveredlabelled;
            answerok = found && prediction == row.at("answer").get<std::string>();
            if (answerok)
                ++correct;
        }
        std::set<std::string> wanted, seen;
        if (row.contains("rule_id"))
            for (const auto &v : row.at("rule_id"))
                wanted.insert(v.is_string() ? v.get<std::string>() : v.dump());
        else if (row.contains("refered_rules"))
            for (const auto &v : row.at("refered_rules")) {
                std::smatch match;
                auto ref = v.get<std::string>();
                if (std::regex_search(ref, match, std::regex("^\\s*([0-9]+)\\s*[.]")))
                    wanted.insert(match[1].str());
            }
        if (!wanted.empty()) {
            if (found && it->second.contains("evidence"))
                for (const auto &h : it->second.at("evidence"))
                    seen.insert(h.at("record_id").get<std::string>());
            ++withrules;
            std::size_t count = 0;
            for (const auto &r : wanted)
                if (seen.count(r))
                    ++count;
            rulecount += wanted.size();
            recalled += count;
            rulesok = count == wanted.size();
            if (rulesok)
                ++allrules;
            if (wanted.size() > 1) {
                ++multi;
                if (rulesok)
                    ++multiall;
            } else {
                ++single;
                if (rulesok)
                    ++singleall;
            }
        }
        if (!found || !answerok || !rulesok)
            failures.push_back(
                {{"question_id", id},
                 {"prediction", prediction},
                 {"answer_correct",
                  row.contains("answer") ? nlohmann::json(answerok) : nlohmann::json(nullptr)},
                 {"all_rules_recalled", rulesok},
                 {"missing", !found}});
    }
    auto ratio = [](std::size_t a, std::size_t b) -> nlohmann::json {
        return b ? nlohmann::json(static_cast<double>(a) / b) : nlohmann::json(nullptr);
    };
    return {{"total", gold.size()},
            {"completed", covered},
            {"labelled", labelled},
            {"correct", correct},
            {"accuracy", ratio(correct, labelled)},
            {"completed_accuracy", ratio(correct, coveredlabelled)},
            {"invalid_answers", invalid},
            {"rule_recall", ratio(recalled, rulecount)},
            {"all_rules_recall", ratio(allrules, withrules)},
            {"single_rule_all_recall", ratio(singleall, single)},
            {"multi_rule_all_recall", ratio(multiall, multi)},
            {"failures", failures}};
}
} // namespace rag
