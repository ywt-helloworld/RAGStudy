#include "Config.hpp"
#include "Document.hpp"
#include "Evaluation.hpp"
#include "Inference.hpp"
#include "Logger.hpp"
#include "Text.hpp"
#include <CLI/CLI.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    try {
        CLI::App app{"Local general-purpose RAG: ingest, retrieve, answer and evaluate"};
        std::vector<std::string> inputs;
        std::string configpath = "config.json", db = "artifacts/index.json", chat, query, questions,
                    output = "artifacts/results.json", evaluate;
        bool lexical = false, retrieve = false, resume = false, choice = false, incremental = false;
        int limit = 0, topk = 0;
        app.add_option("--config", configpath, "Configuration JSON");
        auto build = app.add_option("--build,--ingest", inputs, "Document files or directories")
                         ->expected(-1);
        auto chatopt = app.add_option("--chat", chat, "Interactive chat with an index");
        auto queryopt = app.add_option("--query", query, "Single question");
        auto batchopt = app.add_option("--batch", questions, "Question JSON array");
        auto evalopt = app.add_option("--evaluate", evaluate,
                                      "Evaluate existing results against labelled questions");
        build->excludes(chatopt)->excludes(queryopt)->excludes(batchopt)->excludes(evalopt);
        chatopt->excludes(queryopt)->excludes(batchopt)->excludes(evalopt);
        queryopt->excludes(batchopt)->excludes(evalopt);
        batchopt->excludes(evalopt);
        app.add_option("--db", db, "Persistent index path");
        app.add_option("--output", output, "Results JSON path");
        app.add_flag("--lexical", lexical, "BM25 retrieval only; no embedding model");
        app.add_flag("--retrieve-only", retrieve, "Return evidence without generation");
        app.add_flag("--choice", choice,
                     "Constrain answer to one option (automatic for batch MCQ)");
        app.add_flag("--resume", resume, "Reuse completed rows with matching run signature");
        app.add_flag("--incremental", incremental,
                     "Reuse unchanged embeddings; --build inputs define the complete corpus");
        app.add_option("--limit", limit, "Maximum batch questions (0 = all)")
            ->check(CLI::NonNegativeNumber);
        app.add_option("--top-k", topk, "Override retrieval top_k")->check(CLI::PositiveNumber);
        CLI11_PARSE(app, argc, argv);
        if (argc == 1) {
            std::cout << app.help();
            return 0;
        }
        if (!evaluate.empty()) {
            auto results = nlohmann::json::parse(rag::readText(std::filesystem::u8path(output)));
            auto report = rag::evaluate(evaluate, results);
            rag::writeJson(std::filesystem::u8path(output + ".report.json"), report);
            report.erase("failures");
            std::cout << report.dump(2) << '\n';
            return 0;
        }
        Config cfg(configpath);
        if (topk) {
            cfg.retrieval.top_k = topk;
            cfg.retrieval.candidate_k = std::max(topk, cfg.retrieval.candidate_k);
        }
        rag::Inference inference(cfg);
        auto embeddingIdentity = [&]() {
            return rag::modelIdentity(cfg.models.embedding) +
                   ":mean-window-v1:" + std::to_string(cfg.embedding.n_ctx) + ":" +
                   std::to_string(cfg.embedding.n_batch);
        };
        if (!inputs.empty()) {
            rag::Index index;
            index.chunks = rag::DocumentProcessor(cfg.document).load(inputs);
            if (!lexical) {
                index.embedding_model = embeddingIdentity();
                std::size_t n = 0, reused = 0;
                std::map<std::string, rag::Chunk> old;
                if (incremental && std::filesystem::exists(std::filesystem::u8path(db))) {
                    auto previous = rag::Index::load(db);
                    if (previous.embedding_model == index.embedding_model)
                        for (auto &c : previous.chunks)
                            old.emplace(c.id, std::move(c));
                }
                for (auto &c : index.chunks) {
                    auto it = old.find(c.id);
                    if (it != old.end() && it->second.text == c.text &&
                        it->second.title == c.title) {
                        c.embedding = std::move(it->second.embedding);
                        ++reused;
                    } else
                        c.embedding = inference.embed(c.title + "\n" + c.text);
                    if (++n % 50 == 0)
                        Logger::info("Indexed " + std::to_string(n) + "/" +
                                     std::to_string(index.chunks.size()));
                }
                Logger::info("Reused embeddings: " + std::to_string(reused));
            }
            index.prepare();
            index.save(db);
            std::cout << nlohmann::json({{"chunks", index.chunks.size()},
                                         {"index", db},
                                         {"mode", lexical ? "lexical" : "hybrid"}})
                             .dump(2)
                      << '\n';
            return 0;
        }
        if (!chat.empty())
            db = chat;
        if (query.empty() && questions.empty() && chat.empty())
            throw std::runtime_error("Specify --build, --chat, --query, --batch or --evaluate");
        auto index = rag::Index::load(db);
        if (!lexical) {
            if (index.embedding_model.empty())
                throw std::runtime_error("Lexical index: use --lexical or rebuild with embeddings");
            if (index.embedding_model != embeddingIdentity())
                throw std::runtime_error(
                    "Embedding model or window configuration changed: rebuild index");
        }
        auto ask = [&](const rag::Question &q, bool mcq) {
            auto start = std::chrono::steady_clock::now();
            std::vector<float> vector;
            if (!lexical)
                vector = inference.embed(rag::questionStem(q.text));
            auto hits = index.search(q.text, vector, cfg.retrieval);
            nlohmann::json trace;
            std::string answer;
            if (!retrieve)
                answer = inference.answer(q.text, index, hits, mcq, trace);
            else
                trace["status"] = "retrieved";
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
            return nlohmann::json({{"question_id", q.id},
                                   {"answer", answer},
                                   {"evidence", index.evidence(hits)},
                                   {"trace", trace},
                                   {"elapsed_ms", ms}});
        };
        if (!questions.empty()) {
            auto qs = rag::loadQuestions(questions);
            if (limit > 0 && qs.size() > static_cast<std::size_t>(limit))
                qs.resize(limit);
            std::string genidentity = retrieve ? "none" : rag::modelIdentity(cfg.models.generation);
            auto signature =
                rag::fingerprint(rag::readText(std::filesystem::u8path(db)) +
                                 rag::readText(std::filesystem::u8path(configpath)) + genidentity +
                                 std::to_string(lexical) + std::to_string(retrieve) +
                                 std::to_string(choice) + std::to_string(topk));
            std::map<std::string, nlohmann::json> saved;
            auto acceptPrevious = [&](const nlohmann::json &previous) {
                for (const auto &row : previous) {
                    if (row.value("run_signature", std::string{}) != signature)
                        throw std::runtime_error(
                            "Resume signature mismatch; use a new output path");
                    if (row.value("status", std::string{}) == "ok")
                        saved[row.at("question_id").get<std::string>()] = row;
                }
            };
            if (resume && std::filesystem::exists(std::filesystem::u8path(output))) {
                acceptPrevious(
                    nlohmann::json::parse(rag::readText(std::filesystem::u8path(output))));
            }
            auto journalpath = std::filesystem::u8path(output + ".jsonl");
            if (resume && std::filesystem::exists(journalpath)) {
                std::ifstream journalin(journalpath, std::ios::binary);
                std::string line;
                while (std::getline(journalin, line)) {
                    if (line.empty())
                        continue;
                    // A interrupted final append may be incomplete; completed lines remain usable.
                    auto row = nlohmann::json::parse(line, nullptr, false);
                    if (row.is_discarded()) {
                        if (journalin.eof())
                            break;
                        throw std::runtime_error("Corrupt batch journal");
                    }
                    acceptPrevious(nlohmann::json::array({row}));
                }
            }
            if (!journalpath.parent_path().empty())
                std::filesystem::create_directories(journalpath.parent_path());
            // Compact recovered rows before appending, removing a partial tail after interruption.
            {
                std::ofstream compact(journalpath, std::ios::binary | std::ios::trunc);
                if (!compact)
                    throw std::runtime_error("Cannot open batch journal");
                for (const auto &kv : saved)
                    compact << kv.second.dump() << '\n';
            }
            std::ofstream journal(journalpath, std::ios::binary | std::ios::app);
            auto results = nlohmann::json::array();
            std::size_t n = 0, errors = 0;
            for (const auto &q : qs) {
                nlohmann::json row;
                auto existing = saved.find(q.id);
                if (existing != saved.end() &&
                    existing->second.value("input_hash", std::string{}) == rag::fingerprint(q.text))
                    row = existing->second;
                else {
                    try {
                        row = ask(q, choice || !rag::parseOptions(q.text).empty());
                        row["status"] = "ok";
                    } catch (const std::exception &e) {
                        row = {{"question_id", q.id},
                               {"answer", ""},
                               {"status", "error"},
                               {"error", e.what()},
                               {"evidence", nlohmann::json::array()}};
                        Logger::error("Question " + q.id + ": " + e.what());
                        ++errors;
                    }
                    row["run_signature"] = signature;
                    row["input_hash"] = rag::fingerprint(q.text);
                    journal << row.dump() << '\n';
                    journal.flush();
                    if (!journal)
                        throw std::runtime_error("Batch checkpoint write failed");
                }
                results.push_back(row);
                if ((n + 1) % 100 == 0)
                    rag::writeJson(std::filesystem::u8path(output), results);
                if (++n % 10 == 0)
                    Logger::info("Completed " + std::to_string(n) + "/" +
                                 std::to_string(qs.size()));
            }
            rag::writeJson(std::filesystem::u8path(output), results);
            auto submission = nlohmann::json::array();
            for (const auto &row : results)
                submission.push_back(
                    {{"question_id", row.at("question_id")}, {"answer", row.at("answer")}});
            rag::writeJson(std::filesystem::u8path(output + ".answers.json"), submission);
            auto report = rag::evaluate(questions, results);
            report["run_signature"] = signature;
            report["retrieval_top_k"] = cfg.retrieval.top_k;
            report["retrieval_only"] = retrieve;
            if (retrieve) {
                report["accuracy"] = nullptr;
                report["completed_accuracy"] = nullptr;
                report["correct"] = nullptr;
                report["invalid_answers"] = nullptr;
            }
            rag::writeJson(std::filesystem::u8path(output + ".report.json"), report);
            report.erase("failures");
            std::cout << report.dump(2) << '\n';
            return errors ? 2 : 0;
        }
        if (!query.empty()) {
            auto result = ask({"1", query}, choice);
            std::cout << result.dump(2) << '\n';
            return 0;
        }
        std::string question;
        std::cout << "输入问题，/exit 退出。每轮独立检索。\n";
        while (std::cout << "> " && std::getline(std::cin, question)) {
            if (question == "/exit" || question == "exit")
                break;
            if (question.empty())
                continue;
            try {
                auto result =
                    ask({"chat", question}, choice || !rag::parseOptions(question).empty());
                std::cout << result.dump(2) << '\n';
            } catch (const std::exception &e) {
                Logger::error(e.what());
            }
        }
    } catch (const std::exception &e) {
        Logger::error(e.what());
        return 1;
    }
    return 0;
}
