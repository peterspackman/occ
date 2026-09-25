#pragma once
#include <occ/core/molecule.h>
#include <occ/io/json_cache.h>
#include <occ/qm/wavefunction.h>

namespace occ::driver {
using WavefunctionList = std::vector<occ::qm::Wavefunction>;
using MoleculeList = std::vector<occ::core::Molecule>;

// Each of these keeps its results in `cache` under `<name>.owf.json` (a
// wavefunction) or `<basename>_<i>_monomer_energies.json` (its monomer
// energies), and reuses a cached result only when it was computed the same
// way. Pass an occ::io::FileJsonCache to keep them between runs.

occ::qm::Wavefunction calculate_wavefunction(const occ::core::Molecule &mol,
                                             const std::string &name,
                                             const std::string &energy_model,
                                             bool spherical,
                                             occ::io::JsonCache &cache);

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const MoleculeList &molecules,
                                         const std::string &energy_model,
                                         bool spherical,
                                         occ::io::JsonCache &cache);

/// Compute a wavefunction at an explicit method/basis rather than a CE model name.
occ::qm::Wavefunction
calculate_wavefunction(const occ::core::Molecule &mol, const std::string &name,
                       const std::string &method, const std::string &basis,
                       bool spherical, occ::io::JsonCache &cache);

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const MoleculeList &molecules,
                                         const std::string &method,
                                         const std::string &basis,
                                         bool spherical,
                                         occ::io::JsonCache &cache);

void compute_monomer_energies(const std::string &basename,
                              WavefunctionList &wavefunctions,
                              const std::string &model_name,
                              occ::io::JsonCache &cache);

} // namespace occ::driver
