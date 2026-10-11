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
    int candidate_k = 40;
    int rrf_k = 60;
    int context_chars = 7000;
    float bm25_k1 = 1.2f;
    float bm25_b = 0.75f;
    float vector_weight = 0.25f;
    bool option_queries = true;
};

class Config {
  public:
    explicit Config(const std::string &filename);

    DocumentConfig document;

    EmbeddingConfig embedding;

    GenerationConfig generation;

    ModelConfig models;

    RetrievalConfig retrieval;

  private:
    void validate() const;
};
