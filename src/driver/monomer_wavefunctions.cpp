#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/driver/monomer_wavefunctions.h>
#include <occ/driver/single_point.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/io/occ_input.h>
#include <occ/qm/io/wavefunction_json.h>

namespace occ::driver {

using occ::core::Element;
using occ::core::Molecule;
using occ::qm::Wavefunction;

void compute_monomer_energies(const std::string &basename,
                              WavefunctionList &wavefunctions,
                              const std::string &model_name,
                              occ::io::JsonCache &cache) {
  size_t idx = 0;

  auto model = occ::interaction::ce_model_from_string(model_name);
  occ::interaction::CEModelInteraction interaction(model);
  for (auto &wfn : wavefunctions) {
    const std::string key =
        fmt::format("{}_{}_monomer_energies.json", basename, idx);
    // Cached as {"model": ..., "energy": ...}. The key does not name the
    // model, so a document that names another model, or none, is recomputed.
    bool loaded = false;
    if (const auto cached = cache.load(key)) {
      if (cached->contains("model") && (*cached)["model"] == model.name) {
        occ::log::info("Loading monomer {} energies from {}", idx, key);
        wfn.energy = (*cached)["energy"].get<occ::qm::Energy>();
        loaded = true;
      } else {
        occ::log::warn("Cached monomer energies {} are not for {}; recomputing",
                       key, model.name);
      }
    }
    if (!loaded) {
      occ::log::info("Computing monomer {} energies", idx);
      interaction.compute_monomer_energies(wfn);
      occ::log::info("Caching monomer energies as {}", key);
      nlohmann::json j;
      j["model"] = model.name;
      j["energy"] = wfn.energy;
      cache.store(key, j);
    }
    idx++;
  }
}

namespace {
// The .owf.json cache is keyed only by filename; reuse it only when the stored
// level matches. Legacy caches record method as "SCF" and are trusted as-is.
bool cached_level_matches(const Wavefunction &cached, const std::string &method,
                          const std::string &basis, bool spherical) {
  if (cached.basis.name() != basis || cached.basis.is_pure() != spherical)
    return false;
  if (cached.method != "SCF" && cached.method != method)
    return false;
  return true;
}
} // namespace

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name,
                                    const std::string &method,
                                    const std::string &basis, bool spherical,
                                    occ::io::JsonCache &cache) {
  const std::string key = fmt::format("{}.owf.json", name);
  if (const auto doc = cache.load(key)) {
    auto cached = doc->get<Wavefunction>();
    if (cached_level_matches(cached, method, basis, spherical)) {
      occ::log::info("Loading gas phase wavefunction from {}", key);
      return cached;
    }
    occ::log::warn("Cached wavefunction {} was computed at a different level "
                   "({}/{}, spherical={}); recomputing at {}/{} (spherical={})",
                   key, cached.method, cached.basis.name(),
                   cached.basis.is_pure(), method, basis, spherical);
  }

  occ::io::OccInput input;
  input.method.name = method;
  input.basis.name = basis;
  input.basis.spherical = spherical;
  input.geometry.set_molecule(mol);
  input.electronic.charge = mol.charge();
  input.electronic.multiplicity = mol.multiplicity();
  auto wfn = occ::driver::single_point(input);
  wfn.method = method; // recorded for cache validation

  cache.store(key, wfn);
  return wfn;
}

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name,
                                    const std::string &energy_model,
                                    bool spherical, occ::io::JsonCache &cache) {
  const auto pm = occ::interaction::ce_model_from_string(energy_model);
  return calculate_wavefunction(mol, name, pm.method, pm.basis, spherical,
                                cache);
}

namespace {
void log_molecule(size_t index, const Molecule &m) {
  occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}", index, "sym",
                 "x", "y", "z");
  for (const auto &atom : m.atoms()) {
    occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                   Element(atom.atomic_number).symbol(), atom.x, atom.y, atom.z);
  }
}
} // namespace

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const std::vector<Molecule> &molecules,
                                         const std::string &energy_model,
                                         bool spherical,
                                         occ::io::JsonCache &cache) {
  WavefunctionList wavefunctions;
  size_t index = 0;
  for (const auto &m : molecules) {
    log_molecule(index, m);
    std::string name = fmt::format("{}_{}", basename, index);
    wavefunctions.emplace_back(
        calculate_wavefunction(m, name, energy_model, spherical, cache));
    index++;
  }
  return wavefunctions;
}

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const std::vector<Molecule> &molecules,
                                         const std::string &method,
                                         const std::string &basis,
                                         bool spherical,
                                         occ::io::JsonCache &cache) {
  WavefunctionList wavefunctions;
  size_t index = 0;
  for (const auto &m : molecules) {
    log_molecule(index, m);
    std::string name = fmt::format("{}_{}", basename, index);
    wavefunctions.emplace_back(
        calculate_wavefunction(m, name, method, basis, spherical, cache));
    index++;
  }
  return wavefunctions;
}
} // namespace occ::driver
