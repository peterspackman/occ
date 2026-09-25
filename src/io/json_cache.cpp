#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <occ/io/json_cache.h>
#include <stdexcept>

namespace occ::io {

std::optional<nlohmann::json>
FileJsonCache::load(const std::string &key) const {
  if (!std::filesystem::exists(key))
    return std::nullopt;
  std::ifstream ifs(key);
  if (!ifs)
    throw std::runtime_error(fmt::format("Cannot read cache file '{}'", key));
  return nlohmann::json::parse(ifs);
}

void FileJsonCache::store(const std::string &key, const nlohmann::json &doc) {
  std::ofstream ofs(key);
  if (!ofs)
    throw std::runtime_error(fmt::format("Cannot write cache file '{}'", key));
  ofs << doc;
}

std::string FileJsonCache::location(const std::string &key) const {
  return fmt::format("file {}", key);
}

std::optional<nlohmann::json>
MemoryJsonCache::load(const std::string &key) const {
  const auto it = m_documents.find(key);
  if (it == m_documents.end())
    return std::nullopt;
  return it->second;
}

void MemoryJsonCache::store(const std::string &key, const nlohmann::json &doc) {
  m_documents[key] = doc;
}

std::string MemoryJsonCache::location(const std::string &key) const {
  return fmt::format("memory [{}]", key);
}

} // namespace occ::io
