#include <filesystem>
#include <fmt/os.h>
#include <nlohmann/json.hpp>
#include <occ/interaction/interaction_json.h>
#include <occ/interaction/pair_energy_store.h>

namespace fs = std::filesystem;
using occ::core::Dimer;

namespace occ::interaction {

bool load_dimer_energy(const std::string &filename,
                       CEEnergyComponents &energies) {
  if (!fs::exists(filename))
    return false;
  occ::log::debug("Load dimer energies from {}", filename);
  std::ifstream file(filename);
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  energies = nlohmann::json::parse(line);
  energies.is_computed = true;
  return true;
}

bool write_xyz_dimer(const std::string &filename, const Dimer &dimer,
                     std::optional<CEEnergyComponents> energies) {

  using occ::core::Element;
  auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                               fmt::file::CREATE);
  const auto &pos = dimer.positions();
  const auto &nums = dimer.atomic_numbers();
  output.print("{}\n", nums.rows());
  if (energies) {
    nlohmann::json j = *energies;
    output.print("{}", j.dump());
  }
  output.print("\n");
  for (size_t i = 0; i < nums.rows(); i++) {
    output.print("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                 Element(nums(i)).symbol(), pos(0, i), pos(1, i), pos(2, i));
  }
  return true;
}

bool MemoryPairEnergyStore::save(int id, const core::Dimer &,
                                 const CEEnergyComponents &e) {
  energy_store[id] = e;
  return true;
}

bool MemoryPairEnergyStore::load(int id, const core::Dimer &,
                                 CEEnergyComponents &e) {
  auto it = energy_store.find(id);
  if (it == energy_store.end())
    return false;
  e = it->second;
  e.is_computed = true;
  return true;
}

std::string MemoryPairEnergyStore::dimer_filename(int id,
                                                  const core::Dimer &) const {
  return fmt::format("memory_dimer_{}", id);
}

bool FileSystemPairEnergyStore::save(int id, const core::Dimer &d,
                                     const CEEnergyComponents &e) {
  fs::path parent(base_path);
  if (!fs::exists(parent)) {
    fs::create_directories(parent);
  }

  auto filepath = parent / fs::path(dimer_filename(id, d));

  if (format == Format::JSON) {
    nlohmann::json j;
    j["id"] = id;
    j["energy"] = e;
    std::ofstream file(filepath);
    file << j.dump(2);
    return true;
  } else {
    return write_xyz_dimer(filepath.string(), d, e);
  }
}

bool FileSystemPairEnergyStore::load(int id, const core::Dimer &d,
                                     CEEnergyComponents &e) {
  fs::path parent(base_path);
  auto filepath = parent / fs::path(dimer_filename(id, d));

  if (!fs::exists(filepath))
    return false;

  if (format == Format::JSON) {
    std::ifstream file(filepath);
    nlohmann::json j;
    file >> j;
    e = j["energy"].get<CEEnergyComponents>();
    e.is_computed = true;
    return true;
  } else {
    return load_dimer_energy(filepath.string(), e);
  }
}

std::string
FileSystemPairEnergyStore::dimer_filename(int id, const core::Dimer &) const {
  return fmt::format("dimer_{}.{}", id,
                     format == Format::JSON ? "json" : "xyz");
}

// PairEnergyStore implementation
PairEnergyStore::PairEnergyStore(Kind kind, std::string name)
    : kind(kind), name(std::move(name)) {
  switch (kind) {
  case Kind::Memory:
    store = std::make_unique<MemoryPairEnergyStore>();
    break;
  case Kind::JSON:
    store = std::make_unique<FileSystemPairEnergyStore>(
        this->name, FileSystemPairEnergyStore::Format::JSON);
    break;
  case Kind::XYZ:
    store = std::make_unique<FileSystemPairEnergyStore>(
        this->name, FileSystemPairEnergyStore::Format::XYZ);
    break;
  }
}

bool PairEnergyStore::save(int id, const core::Dimer &d,
                           const CEEnergyComponents &e) {
  return store->save(id, d, e);
}

bool PairEnergyStore::load(int id, const core::Dimer &d,
                           CEEnergyComponents &e) {
  return store->load(id, d, e);
}

std::string PairEnergyStore::dimer_filename(int id,
                                            const core::Dimer &d) const {
  return store->dimer_filename(id, d);
}

} // namespace occ::interaction
