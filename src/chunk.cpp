#include "chunk.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

Chunk::Chunk(std::string text, std::string metadata,
             std::vector<float> embedding)
    : text(std::move(text)),
      metadata(std::move(metadata)),
      embedding(std::move(embedding)) {}

void Chunk::serialize(std::ofstream& out) const {
  if (!out) {
    throw std::runtime_error(
        "Chunk serialize failed: output stream is invalid");
  }

  std::uint64_t text_len = static_cast<std::uint64_t>(text.size());
  out.write(reinterpret_cast<const char*>(&text_len),
            static_cast<std::streamsize>(sizeof(text_len)));
  out.write(text.data(), static_cast<std::streamsize>(text.size()));

  std::uint64_t metadata_len = static_cast<std::uint64_t>(metadata.size());
  out.write(reinterpret_cast<const char*>(&metadata_len),
            static_cast<std::streamsize>(sizeof(metadata_len)));
  out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));

  std::uint64_t emb_dim = static_cast<std::uint64_t>(embedding.size());
  out.write(reinterpret_cast<const char*>(&emb_dim),
            static_cast<std::streamsize>(sizeof(emb_dim)));

  if (!embedding.empty()) {
    out.write(reinterpret_cast<const char*>(embedding.data()),
              static_cast<std::streamsize>(embedding.size() * sizeof(float)));
  }

  if (!out) {
    throw std::runtime_error("Chunk serialize failed: write error");
  }
}

Chunk Chunk::deserialize(std::ifstream& in) {
  if (!in) {
    throw std::runtime_error(
        "Chunk deserialize failed: input stream is invalid");
  }

  std::uint64_t text_len = 0;
  in.read(reinterpret_cast<char*>(&text_len),
          static_cast<std::streamsize>(sizeof(text_len)));

  std::string text;
  text.resize(static_cast<std::size_t>(text_len));
  if (text_len > 0) {
    in.read(text.data(), static_cast<std::streamsize>(text_len));
  }

  std::uint64_t metadata_len = 0;
  in.read(reinterpret_cast<char*>(&metadata_len),
          static_cast<std::streamsize>(sizeof(metadata_len)));

  std::string metadata;
  metadata.resize(static_cast<std::size_t>(metadata_len));
  if (metadata_len > 0) {
    in.read(metadata.data(), static_cast<std::streamsize>(metadata_len));
  }

  std::uint64_t emb_dim = 0;
  in.read(reinterpret_cast<char*>(&emb_dim),
          static_cast<std::streamsize>(sizeof(emb_dim)));

  std::vector<float> embedding;
  embedding.resize(static_cast<std::size_t>(emb_dim));
  if (emb_dim > 0) {
    in.read(reinterpret_cast<char*>(embedding.data()),
            static_cast<std::streamsize>(embedding.size() * sizeof(float)));
  }

  if (!in) {
    throw std::runtime_error("Chunk deserialize failed: read error");
  }

  return Chunk(std::move(text), std::move(metadata), std::move(embedding));
}
