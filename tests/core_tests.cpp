#include "Document.hpp"
#include "Evaluation.hpp"
#include "Index.hpp"
#include "Text.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace rag;
void check(bool ok, const std::string &msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
template <class F> void mustFail(F f, const std::string &msg) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    check(failed, msg);
}
int main() {
    try {
        auto root = std::filesystem::temp_directory_path() / "rag-core-tests";
        std::filesystem::create_directories(root);
        check(clip("甲乙😀丙", 3) == "甲乙😀", "UTF-8 clip");
        mustFail([] { characters(std::string("\xed\xa0\x80")); }, "Reject surrogate");
        auto opts = parseOptions("选择：A. 每天1次 B. 每天2次 C. 不执行 D. 未说明");
        check(opts.size() == 4 && opts[2].label == "C", "Inline options");
        check(parseOptions("A、选项甲\nB、选项乙").size() == 2, "Chinese option delimiter");
        auto doc = root / "docs.json";
        writeJson(doc, nlohmann::json::array(
                           {{{"id", "x"}, {"text", "北站每日开展1次观测，14时前上报。"}},
                            {{"id", "y"}, {"text", "南站负责医疗救护。"}}}));
        auto chunks = DocumentProcessor({8, 2}).load({doc.u8string()});
        check(chunks.size() == 2 && chunks[0].record_id == "x", "Atomic JSON records");
        auto bad = root / "questions.json";
        writeJson(bad, nlohmann::json::array({{{"question_id", "1"},
                                               {"question_text", "哪项正确？"},
                                               {"answer", "A"},
                                               {"rule_id", {"x"}}}}));
        mustFail([&] { DocumentProcessor({8, 2}).load({bad.u8string()}); }, "Do not ingest labels");
        Index index;
        index.chunks = chunks;
        index.prepare();
        RetrievalConfig cfg;
        cfg.top_k = 2;
        auto hits = index.search("北站观测每日上报", {}, cfg);
        check(!hits.empty() && hits[0].index == 0, "Chinese BM25");
        auto file = root / "index.json";
        index.save(file.u8string());
        auto loaded = Index::load(file.u8string());
        check(loaded.chunks.size() == 2, "Round trip");
        index.save(file.u8string());
        auto tamper = nlohmann::json::parse(readText(file));
        tamper["chunks"][0]["text"] = "更改";
        writeJson(file, tamper);
        mustFail([&] { Index::load(file.u8string()); }, "Checksum corruption");
        index.chunks[0].embedding = {1, 0};
        index.chunks[1].embedding = {0, 1};
        index.embedding_model = "test";
        index.prepare();
        mustFail([&] { index.search("北站", {1}, cfg); }, "Dimension check");
        mustFail([&] { index.search("北站", {0, 0}, cfg); }, "Zero vector");
        writeJson(bad, nlohmann::json::array({{{"question_id", "1"},
                                               {"question_text", "选择：A.甲 B.乙"},
                                               {"answer", "A"},
                                               {"rule_id", {"x", "y"}}}}));
        auto qs = loadQuestions(bad.u8string());
        check(qs.size() == 1 && qs[0].text.find("选择") != std::string::npos, "Question adapter");
        auto results =
            nlohmann::json::array({{{"question_id", "1"},
                                    {"answer", "A"},
                                    {"evidence", nlohmann::json::array({{{"record_id", "x"}}})}}});
        auto report = evaluate(bad.u8string(), results);
        auto mismatched = results;
        mismatched[0]["input_hash"] = "other-question";
        mustFail([&] { evaluate(bad.u8string(), mismatched); },
                 "Reject overlapping IDs from another question set");
        check(report["accuracy"] == 1.0 && report["rule_recall"] == 0.5 &&
                  report["all_rules_recall"] == 0.0,
              "Multi-rule metrics");
        auto txt = root / "utf8.txt";
        {
            std::ofstream f(txt, std::ios::binary);
            f << "# 第一章\n甲乙丙丁戊己庚辛壬癸。\n# 第二章\n另一章内容。";
        }
        auto split = DocumentProcessor({8, 2}).load({txt.u8string()});
        check(split.size() >= 3, "Heading split and overlap");
        for (const auto &c : split)
            characters(c.text);
        std::cout << "All core checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
