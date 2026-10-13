#include "mragapp.hpp"

#include <llama.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "embeddingengine.hpp"
#include "generationengine.hpp"
#include "logger.hpp"
#include "vectordatabase.hpp"

namespace {

// 每条 JSON 文档记录直接对应一个检索单元，不做章节识别或文本切分。
std::vector<Chunk> ReadJsonChunks(const std::string& file_path) {
  const auto path = std::filesystem::u8path(file_path);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法读取 JSON 文件: " + file_path);
  }
  nlohmann::json data;
  input >> data;
  if (data.is_object() && data.contains("documents")) {
    auto documents = data.at("documents");
    data = std::move(documents);
  }
  if (data.is_object()) {
    data = nlohmann::json::array({data});
  }
  if (!data.is_array()) {
    throw std::runtime_error(
        "JSON 须为文档对象、对象数组或 {documents:[...]}。");
  }
  std::vector<Chunk> chunks;
  chunks.reserve(data.size());
  std::set<std::string> ids;
  for (std::size_t index = 0; index < data.size(); ++index) {
    const auto& row = data.at(index);
    if (!row.is_object()) {
      throw std::runtime_error("JSON 文档记录须为对象。");
    }
    if (row.contains("question_text") || row.contains("answer") ||
        row.contains("refered_rules")) {
      throw std::runtime_error("题目与标准答案用于评测，不能作为知识库入库。");
    }
    std::string body;
    bool has_body = false;
    for (const char* key : {"text", "content", "rule_text"}) {
      if (row.contains(key)) {
        body = row.at(key).get<std::string>();
        has_body = true;
        break;
      }
    }
    if (!has_body) {
      throw std::runtime_error(
          "JSON 文档记录缺少 text、content 或 rule_text。");
    }
    std::string id = std::to_string(index + 1);
    for (const char* key : {"id", "doc_id", "rule_id"}) {
      if (row.contains(key)) {
        id = row.at(key).is_string() ? row.at(key).get<std::string>()
                                     : row.at(key).dump();
        break;
      }
    }
    if (!ids.insert(id).second) {
      throw std::runtime_error("重复文档编号: " + id);
    }
    if (body.find_first_not_of(" \t\r\n") == std::string::npos) {
      Logger::Warning("跳过空文档记录: ", id);
      continue;
    }
    std::string metadata = path.filename().u8string() + " | 记录 " + id;
    const auto title = row.value("title", std::string{});
    if (!title.empty()) {
      metadata += " | " + title;
    }
    chunks.emplace_back(std::move(body), std::move(metadata));
  }
  return chunks;
}

std::string trimRuntimeCommand(const std::string& s) {
  std::size_t begin = 0;

  while (begin < s.size() &&
         std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
    ++begin;
  }

  std::size_t end = s.size();

  while (end > begin &&
         std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
    --end;
  }

  return s.substr(begin, end - begin);
}

bool parseAdjustCommand(const std::string& line, std::string& key,
                        std::string& value) {
  std::istringstream iss(line);

  std::string command;
  iss >> command;

  if (command != "/adjust") {
    return false;
  }

  if (!(iss >> key)) {
    return false;
  }

  std::getline(iss, value);
  value = trimRuntimeCommand(value);

  return !key.empty() && !value.empty();
}

bool readWholeFile(const std::string& path, std::string& content) {
  std::ifstream in(path, std::ios::binary);

  if (!in.is_open()) {
    return false;
  }

  std::ostringstream oss;
  oss << in.rdbuf();
  content = oss.str();

  return true;
}

bool writeWholeFile(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);

  if (!out.is_open()) {
    return false;
  }

  out << content;

  return static_cast<bool>(out);
}

std::string NormalizeAnswer(std::string text) {
  // 全角 ASCII 及全角空格转为半角，中文句号也允许作为答案结尾。
  std::string normalized;
  for (std::size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i]);
    if (i + 2 < text.size() && (first & 0xf0) == 0xe0) {
      const auto second = static_cast<unsigned char>(text[i + 1]);
      const auto third = static_cast<unsigned char>(text[i + 2]);
      if ((second & 0xc0) == 0x80 && (third & 0xc0) == 0x80) {
        const unsigned code =
            ((first & 0x0f) << 12) | ((second & 0x3f) << 6) | (third & 0x3f);
        if (code >= 0xff01 && code <= 0xff5e) {
          normalized.push_back(static_cast<char>(code - 0xfee0));
          i += 3;
          continue;
        }
        if (code == 0x3000 || code == 0x3002) {
          normalized.push_back(code == 0x3000 ? ' ' : '.');
          i += 3;
          continue;
        }
      }
    }
    normalized.push_back(text[i++]);
  }
  return trimRuntimeCommand(normalized);
}

std::string ExtractAnswer(const std::string& output) {
  const std::string marker = "【最终答案】";
  auto answer = NormalizeAnswer(output);
  const auto start = answer.find(marker);
  if (start != std::string::npos) {
    if (answer.find(marker, start + marker.size()) != std::string::npos) {
      return {};
    }
    answer = trimRuntimeCommand(answer.substr(start + marker.size()));
    if (!answer.empty() && answer.front() == ':') {
      answer = trimRuntimeCommand(answer.substr(1));
    }
  }
  if (!answer.empty() && answer.back() == '.') {
    answer = trimRuntimeCommand(answer.substr(0, answer.size() - 1));
  }
  if (answer.size() != 1) {
    return {};
  }
  const auto choice = static_cast<char>(
      std::toupper(static_cast<unsigned char>(answer.front())));
  return choice >= 'A' && choice <= 'D' ? std::string(1, choice)
                                        : std::string{};
}

std::string ReadId(const nlohmann::json& value) {
  if (!value.is_string() && !value.is_number_integer()) {
    throw std::runtime_error("编号须为字符串或整数。");
  }
  auto id = trimRuntimeCommand(value.is_string() ? value.get<std::string>()
                                                 : value.dump());
  if (id.empty()) {
    throw std::runtime_error("编号不能为空。");
  }
  return id;
}

std::string DocumentId(const Chunk& chunk) {
  const std::string marker = " | 记录 ";
  const auto start = chunk.metadata.find(marker);
  if (start == std::string::npos) {
    throw std::runtime_error("数据库缺少文档原始编号，请重新建库。");
  }
  const auto begin = start + marker.size();
  const auto end = chunk.metadata.find(" | ", begin);
  auto id = trimRuntimeCommand(chunk.metadata.substr(begin, end - begin));
  if (id.empty()) {
    throw std::runtime_error("数据库中的文档编号为空。");
  }
  return id;
}

struct RetrievalMetrics {
  bool hit = false;
  bool all_hit = false;
  double recall = 0.0;
};

RetrievalMetrics ScoreRetrieval(const std::set<std::string>& expected,
                                const std::set<std::string>& retrieved) {
  std::size_t hits = 0;
  for (const auto& id : expected) {
    hits += retrieved.count(id);
  }
  return {hits != 0, !expected.empty() && hits == expected.size(),
          expected.empty() ? 0.0 : static_cast<double>(hits) / expected.size()};
}
}  // namespace

MragApp::BackendGuard::BackendGuard() { llama_backend_init(); }

MragApp::BackendGuard::~BackendGuard() { llama_backend_free(); }

MragApp::MragApp(const AppConfig& config_) : backend(), config(config_) {}

bool MragApp::RunTest(const std::string& questions_path,
                      const std::string& db_path) {
  try {
    namespace fs = std::filesystem;
    using nlohmann::json;
    const auto output_path = fs::absolute("test/result");
    if (fs::weakly_canonical(fs::u8path(questions_path)) == output_path ||
        fs::weakly_canonical(fs::u8path(db_path)) == output_path) {
      throw std::runtime_error("结果文件不能覆盖题集或数据库。");
    }
    std::ifstream input(fs::u8path(questions_path), std::ios::binary);
    if (!input) {
      throw std::runtime_error("无法读取题集: " + questions_path);
    }
    json questions;
    input >> questions;
    if (!questions.is_array() || questions.empty()) {
      throw std::runtime_error("题集须为非空 JSON 数组。");
    }
    VectorDatabase database;
    if (!database.loadFromDisk(db_path) || database.empty()) {
      throw std::runtime_error("无法读取非空知识库: " + db_path);
    }
    std::set<std::string> corpus_ids;
    const auto dimension = database.documents().front().embedding.size();
    for (const auto& document : database.documents()) {
      if (!corpus_ids.insert(DocumentId(document)).second) {
        throw std::runtime_error("知识库存在重复编号，评测要求编号唯一。");
      }
      if (dimension == 0 || document.embedding.size() != dimension) {
        throw std::runtime_error("知识库向量为空或维度不一致，请重新建库。");
      }
    }
    std::set<std::string> question_ids;
    for (auto& question : questions) {
      const auto id = ReadId(question.at("question_id"));
      if (!question_ids.insert(id).second ||
          trimRuntimeCommand(question.at("question_text").get<std::string>())
              .empty()) {
        throw std::runtime_error("题号重复或题干为空: " + id);
      }
      const auto answer =
          NormalizeAnswer(question.at("answer").get<std::string>());
      if (answer.size() != 1 || answer.front() < 'A' || answer.front() > 'D') {
        throw std::runtime_error("标准答案须为 A/B/C/D: " + id);
      }
      const auto& references = question.at("rule_id");
      if (!references.is_array() || references.empty()) {
        throw std::runtime_error("标准规则须为非空数组: " + id);
      }
      std::set<std::string> ids;
      for (const auto& reference : references) {
        const auto rule_id = ReadId(reference);
        if (!corpus_ids.count(rule_id)) {
          throw std::runtime_error("题 " + id +
                                   " 的标准规则不在知识库中: " + rule_id);
        }
        ids.insert(rule_id);
      }
      question["question_id"] = id;
      question["answer"] = answer;
      question["rule_id"] = ids;
    }

    AppConfig test_config = config;
    test_config.setRetrievalTopK(5)->setTemperature(0.0);
    std::cout << "题集检查通过，共 " << questions.size()
              << " 题。正在加载模型..." << std::endl;
    EmbeddingEngine embedding(test_config);
    if (embedding.generateEmbedding("向量维度检查").size() != dimension) {
      throw std::runtime_error(
          "当前嵌入模型与数据库向量维度不符，请重新建库。");
    }
    GenerationEngine generation(test_config);
    fs::create_directories(output_path.parent_path());
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    const auto write = [&](const json& record) {
      output << record.dump(-1, ' ', false, json::error_handler_t::replace)
             << '\n';
      output.flush();
    };
    write({{"type", "run"},
           {"protocol", "single-choice-top5-v1"},
           {"questions", fs::absolute(fs::u8path(questions_path)).u8string()},
           {"database", fs::absolute(fs::u8path(db_path)).u8string()},
           {"total", questions.size()},
           {"configuration", test_config.ToJson()}});
    Logger::Info("开始评测，题数=", questions.size(), "，输出=test/result");

    std::size_t correct = 0, hit = 0, all_hit = 0, failed = 0, invalid = 0;
    std::size_t correct_hit = 0, correct_miss = 0, wrong_hit = 0,
                wrong_miss = 0;
    double recall = 0.0;
    for (std::size_t index = 0; index < questions.size(); ++index) {
      const auto& question = questions.at(index);
      const auto expected = question.at("rule_id").get<std::set<std::string>>();
      json record = {{"type", "question"},
                     {"question_id", question.at("question_id")},
                     {"question_text", question.at("question_text")},
                     {"answer", question.at("answer")},
                     {"rule_id", expected},
                     {"predicted_answer", ""},
                     {"raw_output", ""},
                     {"prompt", ""},
                     {"top5", json::array()},
                     {"status", "ok"},
                     {"error", ""}};
      std::set<std::string> retrieved_ids;
      std::string stage = "retrieval_failed";
      try {
        const auto query = question.at("question_text").get<std::string>();
        const auto vector = embedding.generateEmbedding(query);
        if (vector.size() != dimension) {
          throw std::runtime_error("问题向量为空或维度不一致。");
        }
        const auto results = database.hybridSearch(query, vector, 5);
        for (const auto& result : results) {
          if (!result.chunk || !std::isfinite(result.score)) {
            throw std::runtime_error("检索结果为空指针或分数无效。");
          }
          const auto id = DocumentId(*result.chunk);
          retrieved_ids.insert(id);
          record["top5"].push_back({{"rank", record["top5"].size() + 1},
                                    {"rule_id", id},
                                    {"score", result.score}});
        }
        stage = "generation_failed";
        const auto generated = generation.Generate(query, results, false, true);
        record["raw_output"] = generated.text;
        record["prompt"] = generated.prompt;
        record["error"] = generated.error;
        if (!generated.error.empty()) {
          record["status"] = stage;
        } else {
          const auto answer = ExtractAnswer(generated.text);
          record["predicted_answer"] = answer;
          if (answer.empty()) {
            record["status"] = "invalid_answer";
          }
        }
      } catch (const std::exception& error) {
        record["status"] = stage;
        record["error"] = error.what();
        if (stage == "retrieval_failed") {
          retrieved_ids.clear();
          record["top5"] = json::array();
        }
      }
      const auto metrics = ScoreRetrieval(expected, retrieved_ids);
      const bool answer_correct =
          record["status"] == "ok" &&
          record["predicted_answer"] == question.at("answer");
      record["answer_correct"] = answer_correct;
      record["hit_at_5"] = metrics.hit;
      record["all_hit_at_5"] = metrics.all_hit;
      record["recall_at_5"] = metrics.recall;
      write(record);
      correct += answer_correct;
      hit += metrics.hit;
      all_hit += metrics.all_hit;
      recall += metrics.recall;
      failed += record["status"] != "ok";
      invalid += record["status"] == "invalid_answer";
      correct_hit += answer_correct && metrics.hit;
      correct_miss += answer_correct && !metrics.hit;
      wrong_hit += !answer_correct && metrics.hit;
      wrong_miss += !answer_correct && !metrics.hit;
      Logger::Info(
          "评测题完成，题号=", question.at("question_id").get<std::string>(),
          "，状态=", record.at("status").get<std::string>());
      std::cout << '[' << index + 1 << '/' << questions.size() << "] 题 "
                << question.at("question_id").get<std::string>() << "：答案"
                << (answer_correct ? "正确" : "错误") << "，Top5 "
                << (metrics.all_hit ? "全部命中"
                    : metrics.hit   ? "部分命中"
                                    : "未命中")
                << "，状态=" << record.at("status").get<std::string>()
                << std::endl;
    }
    const double total = static_cast<double>(questions.size());
    const json summary = {{"type", "summary"},
                          {"complete", true},
                          {"total", questions.size()},
                          {"correct", correct},
                          {"hit_at_5_count", hit},
                          {"all_hit_at_5_count", all_hit},
                          {"accuracy", correct / total},
                          {"hit_at_5", hit / total},
                          {"all_hit_at_5", all_hit / total},
                          {"recall_at_5", recall / total},
                          {"failed_or_invalid", failed},
                          {"invalid_answers", invalid},
                          {"correct_hit", correct_hit},
                          {"correct_miss", correct_miss},
                          {"wrong_hit", wrong_hit},
                          {"wrong_miss", wrong_miss}};
    write(summary);
    std::cout << "评测完成，结果已写入 test/result\n"
              << summary.dump(2) << std::endl;
    Logger::Info("评测完成，题数=", questions.size(), "，正确=", correct,
                 "，失败或无效=", failed);
    return true;
  } catch (const std::exception& error) {
    Logger::Error("评测失败：", error.what());
    return false;
  }
}

bool MragApp::buildKnowledgeBase(const std::string& jsonPath,
                                 const std::string& dbPath) {
  try {
    Logger::Info("开始建库，文档=", jsonPath, "，输出=", dbPath);
    std::cout << "正在处理文档：" << jsonPath << std::endl;

    std::vector<Chunk> chunks = ReadJsonChunks(jsonPath);

    if (chunks.empty()) {
      Logger::Error("文档处理失败：没有生成任何文本块。");
      return false;
    }

    std::cout << "JSON 读取完成，共得到 " << chunks.size() << " 条文档记录。"
              << std::endl;
    Logger::Info("文档解析完成: ", jsonPath, "，文本块数=", chunks.size());

    std::cout << "正在生成向量，请稍候..." << std::endl;

    EmbeddingEngine embeddingEngine(config);
    embeddingEngine.embedChunks(chunks);

    std::cout << "向量生成完成。" << std::endl;

    VectorDatabase database;
    database.insert(chunks);

    if (!database.saveToDisk(dbPath)) {
      Logger::Error("向量数据库保存失败：", dbPath);
      return false;
    }

    std::cout << "知识库构建完成，已保存到：" << dbPath << std::endl;
    Logger::Info("建库完成: ", dbPath);

    return true;

  } catch (const std::exception& e) {
    Logger::Error("构建知识库失败：", e.what());
    return false;
  }
}

bool MragApp::buildKnowledgeBase(const std::vector<std::string>& jsonPaths,
                                 const std::string& dbPath) {
  if (jsonPaths.empty()) {
    Logger::Error("没有提供任何 JSON 文件，无法构建知识库。\n");
    return false;
  }

  try {
    Logger::Info("开始多文档建库，文档数=", jsonPaths.size(),
                 "，输出=", dbPath);

    std::vector<Chunk> allChunks;

    for (const std::string& jsonPath : jsonPaths) {
      std::cout << "正在处理文档：" << jsonPath << std::endl;

      std::vector<Chunk> chunks = ReadJsonChunks(jsonPath);

      if (chunks.empty()) {
        Logger::Error("JSON 文件 ", jsonPath, " 没有生成任何文本块，已跳过：");
        return false;
      }
      std::cout << "JSON 读取完成，共得到 " << chunks.size() << " 条文档记录。"
                << std::endl;
      Logger::Info("文档解析完成: ", jsonPath, "，文本块数=", chunks.size());

      allChunks.insert(allChunks.end(), std::make_move_iterator(chunks.begin()),
                       std::make_move_iterator(chunks.end()));
    }

    if (allChunks.empty()) {
      Logger::Error("所有 JSON 文件都没有有效文档记录。");
      return false;
    }

    std::cout << "所有文档处理完成，共生成 " << allChunks.size()
              << " 条文档记录。" << std::endl;

    std::cout << "正在生成向量，请稍候..." << std::endl;

    EmbeddingEngine embeddingEngine(config);
    embeddingEngine.embedChunks(allChunks);

    std::cout << "向量生成完成。" << std::endl;

    VectorDatabase database;
    database.insert(allChunks);

    if (!database.saveToDisk(dbPath)) {
      Logger::Error("向量数据库保存失败：", dbPath);
      return false;
    }

    std::cout << "多文档知识库构建完成，已保存到：" << dbPath << std::endl;
    Logger::Info("多文档建库完成: ", dbPath);

    return true;

  } catch (const std::exception& e) {
    Logger::Error("多文档构建知识库失败：", e.what());
    return false;
  }
}

bool MragApp::chatLoop(const std::string& dbPath) {
  try {
    Logger::Info("开始问答，数据库=", dbPath);
    VectorDatabase database;

    if (!database.loadFromDisk(dbPath)) {
      Logger::Error("向量数据库加载失败：", dbPath);
      return false;
    }

    if (database.empty()) {
      Logger::Error("向量数据库为空，无法进行问答。");
      return false;
    }

    std::cout << "知识库加载完成，共 " << database.size() << " 条文档记录。"
              << std::endl;

    std::cout << "正在加载模型，请稍候..." << std::endl;

    std::unique_ptr<EmbeddingEngine> embeddingEngine =
        std::make_unique<EmbeddingEngine>(config);

    std::unique_ptr<GenerationEngine> generationEngine =
        std::make_unique<GenerationEngine>(config);

    std::cout << "\nhello，我是你的知识库 rag 助手！\n";
    std::cout << "有问题，尽管问！\n";
    std::cout << "输入 exit、quit 或 退出 可以结束程序。" << std::endl;

    while (true) {
      std::cout << "\n问题> ";

      std::string query;
      if (!std::getline(std::cin, query)) {
        std::cout << std::endl;
        break;
      }

      if (query.empty()) {
        continue;
      }

      if (query == "exit" || query == "quit" || query == "退出" ||
          query == "q") {
        break;
      }

      if (query == "/config") {
        config.printConfig(std::cout);
        continue;
      }

      if (query.rfind("/adjust", 0) == 0) {
        std::string key;
        std::string value;

        if (!parseAdjustCommand(query, key, value)) {
          Logger::Error("adjust命令正确的用法是：/adjust <配置项> <新值>\n");
          Logger::Error("示例：\n/adjust generation.temperature 0.3\n");
          Logger::Error("/adjust retrieval.top_k 5\n");
          Logger::Error("/adjust models.generation ./models/llm/new.gguf\n");
          continue;
        }

        const std::string configPath = "config.json";

        // 备份当前内存中的配置对象
        AppConfig oldConfig = config;

        // 备份当前 config.json 文件内容
        std::string oldConfigFileContent;
        const bool oldConfigFileExists =
            readWholeFile(configPath, oldConfigFileContent);

        try {
          if (!config.adjustConfig(key, value)->saveToFile(configPath)) {
            config = oldConfig;
            if (oldConfigFileExists) {
              writeWholeFile(configPath, oldConfigFileContent);
            }
            Logger::Error("保存配置失败，已恢复旧配置。\n");
            continue;
          }
        } catch (const std::exception& error) {
          config = oldConfig;
          Logger::Error("运行时配置修改失败：", error.what(), '\n');
          continue;
        }

        try {
          std::cout << "正在根据新配置重新加载模型，请稍候...\n";

          auto newEmbeddingEngine = std::make_unique<EmbeddingEngine>(config);

          auto newGenerationEngine = std::make_unique<GenerationEngine>(config);

          // 只有两个新模型都加载成功，才真正替换旧模型
          embeddingEngine = std::move(newEmbeddingEngine);
          generationEngine = std::move(newGenerationEngine);

          std::cout << "运行时配置热切换成功。\n";
          Logger::Info("运行时配置热切换成功");
          config.printConfig(std::cout);

        } catch (const std::exception& e) {
          Logger::Error("配置已保存，但模型重新加载失败：", e.what(), "\n");

          // 回滚内存中的 config 对象
          config = oldConfig;

          // 回滚 config.json 文件
          if (oldConfigFileExists) {
            if (!writeWholeFile(configPath, oldConfigFileContent)) {
              Logger::Error(
                  "警告：恢复旧 config.json 失败，请手动检查配置文件。\n");
            } else {
              Logger::Error("已恢复旧 config.json。\n");
            }
          } else {
            std::remove(configPath.c_str());
            Logger::Error("已删除本次新生成的 config.json。\n");
          }

          Logger::Error(
              "当前仍保留旧模型继续运行。请检查模型路径或参数后再次 "
              "/adjust。\n");
        }

        continue;
      }

      if (query == "help" || query == "h" || query == "帮助") {
        std::cout << "\nhello，我是你的知识库 rag 助手！\n";
        std::cout << "  直接输入你想询问的问题，比如：\n";
        std::cout << "  设备借用需要哪些步骤？\n";
        std::cout << "  或者：这份文档有哪些注意事项？\n";
        std::cout << "  设备故障时应该联系谁？\n";
        std::cout << "\n有问题，尽管问！\n";
        continue;
      }

      std::vector<float> queryEmbedding =
          embeddingEngine->generateEmbedding(query);

      if (queryEmbedding.empty()) {
        Logger::Error("问题向量生成失败，请换一个问题重试。");
        continue;
      }

      // // 1. 纯向量版本
      // std::vector<SearchResult> results = database.search(queryEmbedding,
      // static_cast<std::size_t>(config.retrievalTopK()));

      // 2. 关键词增强版本
      std::vector<SearchResult> hybridResults = database.hybridSearch(
          query, queryEmbedding,
          static_cast<std::size_t>(config.retrievalTopK()));
      Logger::Info("检索完成，问题字节数=", query.size(),
                   "，命中文本块数=", hybridResults.size());

      // std::cout << "==================== 版本一：纯向量检索
      // ====================\n";

      // if (results.empty()) {
      //     std::cout << "纯向量检索没有找到相关片段。" << std::endl;
      // } else {
      //     generationEngine->generateStream(query, results);
      // }

      // std::cout << "\n\n";
      // std::cout << "==================== 版本二：关键词增强检索
      // ====================\n";

      if (hybridResults.empty()) {
        Logger::Warning("未检索到相关片段");
        std::cout << "关键词增强检索没有找到相关片段。" << std::endl;
      } else {
        generationEngine->generateStream(query, hybridResults);
      }

      std::cout << "\n";
    }

    std::cout << "聊天结束。" << std::endl;
    Logger::Info("问答结束");
    return true;

  } catch (const std::exception& e) {
    Logger::Error("问答过程失败：", e.what());
    return false;
  }
}
