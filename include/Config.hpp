#pragma once

#include <string>


struct DocumentConfig {
    int chunk_size = 500;
    int overlap_size = 50;
};


struct EmbeddingConfig {
    int n_batch = 512;
    int n_ctx = 512;
    int n_ubatch = 512;
};


struct GenerationConfig {
    int max_output_tokens = 512;

    int n_batch = 4096;
    int n_ctx = 4096;
    int n_ubatch = 512;

    float temperature = 0.7f;

    int top_k = 40;

    float top_p = 0.9f;
};


struct ModelConfig {
    std::string embedding;
    std::string generation;

    int n_gpu_layers = 99;
};


struct RetrievalConfig {
    int top_k = 5;
};


class Config {
public:

    explicit Config(const std::string& filename);

    DocumentConfig document;

    EmbeddingConfig embedding;

    GenerationConfig generation;

    ModelConfig models;

    RetrievalConfig retrieval;

private:

    void validate() const;
};

