#pragma once
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>

namespace occ::main {
using WavefunctionList = std::vector<occ::qm::Wavefunction>;
using MoleculeList = std::vector<occ::core::Molecule>;

occ::qm::Wavefunction calculate_wavefunction(const occ::core::Molecule &mol,
                                             const std::string &name,
                                             const std::string &energy_model,
					     bool spherical);

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const MoleculeList &molecules,
                                         const std::string &energy_model,
					 bool spherical);

void compute_monomer_energies(const std::string &basename,
                              WavefunctionList &wavefunctions,
                              const std::string &model_name);

} // namespace occ::main
