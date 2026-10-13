#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

struct Chunk {
  std::string text;
  std::string metadata;
  std::vector<float> embedding;

  Chunk() = default;
  Chunk(std::string text, std::string metadata,
        std::vector<float> embedding = {});
  Chunk(const Chunk& other) = default;
  Chunk(Chunk&& other) noexcept = default;
  Chunk& operator=(const Chunk& other) = default;
  Chunk& operator=(Chunk&& other) noexcept = default;
  ~Chunk() = default;

  void serialize(std::ofstream& out) const;
  static Chunk deserialize(std::ifstream& in);
};
