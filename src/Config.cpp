#include "Config.hpp"
#include "Logger.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>


using json = nlohmann::json;


namespace {

[[noreturn]] void throwConfigError(const std::string& message) {
    Logger::error(message);
    throw std::runtime_error(message);
}


template<typename T>
void loadValue(const json& data, const std::string& key, T& value, const std::string& path) {
    if(data.contains(key))
        value = data[key].get<T>();
    else
        Logger::warning("Missing " + path + ", using default value: " + std::to_string(value));
}


void requireValue(const json& data, const std::string& key, std::string& value, const std::string& path) {
    if(data.contains(key))
        value = data[key].get<std::string>();
    else
        throwConfigError("Missing required config: " + path);
}

}


Config::Config(const std::string& filename) {
    Logger::info("Loading config file: " + filename);


    std::ifstream file(filename);

    if(!file)
        throwConfigError("Cannot open config file: " + filename);


    json data;
    file >> data;


    if(data.contains("document")) {
        auto& document_json = data["document"];

        loadValue(document_json, "chunk_size", document.chunk_size, "document.chunk_size");

        loadValue(document_json, "overlap_size", document.overlap_size, "document.overlap_size");
    }
    else {
        Logger::warning("Missing document config, using default values");
    }


    if(data.contains("embedding")) {
        auto& embedding_json = data["embedding"];

        loadValue(embedding_json, "n_batch", embedding.n_batch, "embedding.n_batch");

        loadValue(embedding_json, "n_ctx", embedding.n_ctx, "embedding.n_ctx");

        loadValue(embedding_json, "n_ubatch", embedding.n_ubatch, "embedding.n_ubatch");
    }
    else {
        Logger::warning("Missing embedding config, using default values");
    }


    if(data.contains("generation")) {
        auto& generation_json = data["generation"];

        loadValue(generation_json, "max_output_tokens", generation.max_output_tokens, "generation.max_output_tokens");

        loadValue(generation_json, "n_batch", generation.n_batch, "generation.n_batch");

        loadValue(generation_json, "n_ctx", generation.n_ctx, "generation.n_ctx");

        loadValue(generation_json, "n_ubatch", generation.n_ubatch, "generation.n_ubatch");

        loadValue(generation_json, "temperature", generation.temperature, "generation.temperature");

        loadValue(generation_json, "top_k", generation.top_k, "generation.top_k");

        loadValue(generation_json, "top_p", generation.top_p, "generation.top_p");
    }
    else {
        Logger::warning("Missing generation config, using default values");
    }


    if(data.contains("models")) {
        auto& models_json = data["models"];

        requireValue(models_json, "embedding", models.embedding, "models.embedding");

        requireValue(models_json, "generation", models.generation, "models.generation");

        loadValue(models_json, "n_gpu_layers", models.n_gpu_layers, "models.n_gpu_layers");
    }
    else {
        throwConfigError("Missing required models config");
    }


    if(data.contains("retrieval")) {
        auto& retrieval_json = data["retrieval"];

        loadValue(retrieval_json, "top_k", retrieval.top_k, "retrieval.top_k");
    }
    else {
        Logger::warning("Missing retrieval config, using default values");
    }


    validate();


    Logger::info("Config loaded successfully");
    Logger::info("Generation model: " + models.generation);
    Logger::info("Embedding model: " + models.embedding);
    Logger::info("Retrieval top_k: " + std::to_string(retrieval.top_k));
}


void Config::validate() const {
    Logger::info("Validating configuration");


    if(document.chunk_size <= 0)
        throwConfigError("chunk_size must be positive");


    if(document.overlap_size < 0)
        throwConfigError("overlap_size cannot be negative");


    if(document.overlap_size >= document.chunk_size)
        throwConfigError("overlap_size must be smaller than chunk_size");


    if(embedding.n_batch <= 0 || embedding.n_ctx <= 0 || embedding.n_ubatch <= 0)
        throwConfigError("Invalid embedding configuration");


    if(generation.max_output_tokens <= 0)
        throwConfigError("max_output_tokens must be positive");


    if(generation.temperature < 0 || generation.temperature > 2)
        throwConfigError("temperature must be between 0 and 2");


    if(generation.top_k <= 0)
        throwConfigError("generation top_k must be positive");


    if(generation.top_p <= 0 || generation.top_p > 1)
        throwConfigError("top_p must be between 0 and 1");


    if(models.embedding.empty())
        throwConfigError("embedding model path is empty");


    if(models.generation.empty())
        throwConfigError("generation model path is empty");


    if(retrieval.top_k <= 0)
        throwConfigError("retrieval top_k must be positive");


    Logger::info("Configuration validation passed");
}