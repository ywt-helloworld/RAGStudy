#include "Inference.hpp"
#include "Text.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

// Optional integration checks; real GGUF files are needed, so this is not a default CTest.
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Usage: rag_model_checks <config.json>");
        Config config(argv[1]);
        rag::Inference inference(config);
        rag::Index index;
        rag::Chunk fault;
        fault.id = "fault";
        fault.source = "handbook.json";
        fault.record_id = "fault";
        fault.title = "故障报告";
        fault.text = "设备发现故障时应停止使用，向运维人员报告资产编号和故障现象。禁止自行拆卸或继"
                     "续通电测试。";
        fault.embedding = inference.embed(fault.text);
        double norm = 0;
        for (float x : fault.embedding) {
            if (!std::isfinite(x))
                throw std::runtime_error("Non-finite embedding");
            norm += x * x;
        }
        if (std::abs(norm - 1) > 0.001)
            throw std::runtime_error("Embedding is not normalized");
        index.chunks.push_back(fault);
        index.embedding_model = "integration-check";
        index.prepare();
        std::string query = "设备发生故障时，哪项正确？\n选择：A.自行拆卸检查 B.继续通电测试 "
                            "C.停止使用并向运维报告 D.不需要报告";
        auto hits = index.search(query, {}, config.retrieval);
        nlohmann::json trace;
        auto choice = inference.answer(query, index, hits, true, trace);
        if (choice != "C")
            throw std::runtime_error("Simple choice regression: expected C, received " + choice);
        auto answer = inference.answer("发现设备故障后，应该怎样处理？", index, hits, false, trace);
        if (answer.empty() || answer.find("<|") != std::string::npos ||
            answer.find("<think>") != std::string::npos)
            throw std::runtime_error("Raw protocol or reasoning markers leaked into answer");
        rag::characters(answer);
        std::cout << "Choice: " << choice << "\nAnswer: " << answer
                  << "\nLocal model checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
