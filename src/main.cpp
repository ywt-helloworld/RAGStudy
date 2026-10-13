#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <llama.h>

#include <CLI11.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "logger.hpp"
#include "mragapp.hpp"

static void silentLogCallback(ggml_log_level level, const char* text,
                              void* user_data) noexcept {
  (void)user_data;
  try {
    if (text != nullptr && level == GGML_LOG_LEVEL_ERROR) {
      Logger::Log(LogLevel::kError, text);
    } else if (text != nullptr && level == GGML_LOG_LEVEL_WARN) {
      Logger::Log(LogLevel::kWarning, text);
    }
  } catch (...) {
  }
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  llama_log_set(silentLogCallback, nullptr);

  try {
    Logger::Init("logs/mrag.log");
    Logger::Info("程序启动");
    CLI::App cli{"应急RAG"};
    cli.failure_message([](const CLI::App* app, const CLI::Error& error) {
      Logger::Log(LogLevel::kError, error.what());
      return CLI::FailureMessage::simple(app, error);
    });
    bool show_config = false;
    std::vector<std::string> adjustment;
    std::vector<std::string> documents;
    std::vector<std::string> test_files;
    std::string database;

    auto* modes = cli.add_option_group("操作", "一次选择一个操作");
    modes->require_option(0, 1);
    modes->add_flag("--config", show_config, "显示 config.json 配置")
        ->disable_flag_override();
    modes->add_option("--adjust", adjustment, "修改并保存配置")
        ->expected(2)
        ->type_name("KEY VALUE");
    modes
        ->add_option("--ingest", documents,
                     "JSON 规则/文档建库，最后一个参数为输出数据库")
        ->expected(2, -1)
        ->type_name("DOCUMENT... DATABASE");
    modes->add_option("--chat", database, "使用已有数据库进行问答")
        ->type_name("DATABASE");
    modes
        ->add_option("--test", test_files, "运行单选题集，结果写入 test/result")
        ->expected(2)
        ->type_name("QUESTIONS DATABASE");

    CLI11_PARSE(cli, argc, cli.ensure_utf8(argv));
    if (modes->count_all() == 0) {
      std::cout << cli.help();
      return 0;
    }

    const std::string config_path = "config.json";
    auto config = std::make_unique<AppConfig>(config_path);
    if (show_config) {
      config->printConfig(std::cout);
      return 0;
    }
    if (!adjustment.empty()) {
      if (!config->adjustConfig(adjustment[0], adjustment[1])
               ->saveToFile(config_path)) {
        return 1;
      }
      std::cout << "修改完成，";
      config->printConfig(std::cout);
      return 0;
    }

    MragApp app(*config);
    if (!test_files.empty()) {
      return app.RunTest(test_files[0], test_files[1]) ? 0 : 1;
    }
    if (!documents.empty()) {
      const std::string output_database = documents.back();
      documents.pop_back();
      if (documents.size() == 1) {
        return app.buildKnowledgeBase(documents.front(), output_database) ? 0
                                                                          : 1;
      }
      return app.buildKnowledgeBase(documents, output_database) ? 0 : 1;
    }
    return app.chatLoop(database) ? 0 : 1;
  } catch (const std::exception& error) {
    Logger::Error("程序运行失败：", error.what());
    return 1;
  }
}
