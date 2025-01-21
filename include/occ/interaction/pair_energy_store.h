#pragma once
#include <memory>
#include <occ/core/dimer.h>
#include <occ/interaction/pair_energy.h>
#include <string>
#include <unordered_map>

namespace occ::interaction {

class PairEnergyStoreBase {
public:
  virtual ~PairEnergyStoreBase() = default;
  virtual bool save(int id, const core::Dimer &d,
                    const CEEnergyComponents &e) = 0;
  virtual bool load(int id, const core::Dimer &d, CEEnergyComponents &e) = 0;
  virtual std::string dimer_filename(int id, const core::Dimer &d) const = 0;
};

class MemoryPairEnergyStore : public PairEnergyStoreBase {
public:
  bool save(int id, const core::Dimer &d, const CEEnergyComponents &e) override;
  bool load(int id, const core::Dimer &d, CEEnergyComponents &e) override;
  std::string dimer_filename(int id, const core::Dimer &d) const override;

private:
  std::unordered_map<int, CEEnergyComponents> energy_store;
};

class FileSystemPairEnergyStore : public PairEnergyStoreBase {
public:
  enum class Format { JSON, XYZ };
  explicit FileSystemPairEnergyStore(std::string path,
                                     Format format = Format::XYZ)
      : base_path(std::move(path)), format(format) {}

  bool save(int id, const core::Dimer &d, const CEEnergyComponents &e) override;
  bool load(int id, const core::Dimer &d, CEEnergyComponents &e) override;
  std::string dimer_filename(int id, const core::Dimer &d) const override;

private:
  std::string base_path;
  Format format;
};

class PairEnergyStore {
public:
  enum class Kind { JSON, XYZ, Memory };

  explicit PairEnergyStore(Kind kind = Kind::XYZ, std::string name = "");

  bool save(int id, const core::Dimer &d, const CEEnergyComponents &e);
  bool load(int id, const core::Dimer &d, CEEnergyComponents &e);
  std::string dimer_filename(int id, const core::Dimer &d) const;

  Kind kind{Kind::XYZ};
  std::string name;

private:
  std::unique_ptr<PairEnergyStoreBase> store;
};

} // namespace occ::interaction
