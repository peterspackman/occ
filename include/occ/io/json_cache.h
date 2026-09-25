#pragma once
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>

namespace occ::io {

/// Somewhere to keep expensive intermediate results (monomer energies,
/// wavefunctions, solvation surfaces) as JSON documents, so a repeat
/// calculation can skip them.
///
/// A key names one document. The file backend uses it as the path, so the
/// keys are the cache filenames the calculations have always written; the
/// memory backend only needs them to be distinct. Checking that a cached
/// document still matches the calculation (model, level of theory, ...) is
/// the caller's job: the cache just stores what it is given.
class JsonCache {
public:
  virtual ~JsonCache() = default;

  /// The document stored under `key`, or nothing if there is none. A
  /// document that exists but cannot be parsed is an error, not a miss.
  virtual std::optional<nlohmann::json> load(const std::string &key) const = 0;

  virtual void store(const std::string &key, const nlohmann::json &doc) = 0;
};

/// Documents on disk, one file per key. Persists between runs.
class FileJsonCache final : public JsonCache {
public:
  std::optional<nlohmann::json> load(const std::string &key) const override;
  void store(const std::string &key, const nlohmann::json &doc) override;
};

/// Documents held for the lifetime of the object; nothing touches the disk.
/// For tests, and for callers that run many calculations in one process.
class MemoryJsonCache final : public JsonCache {
public:
  std::optional<nlohmann::json> load(const std::string &key) const override;
  void store(const std::string &key, const nlohmann::json &doc) override;

  size_t size() const { return m_documents.size(); }

private:
  std::unordered_map<std::string, nlohmann::json> m_documents;
};

} // namespace occ::io
