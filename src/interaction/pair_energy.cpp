#include <ankerl/unordered_dense.h>
#include <fmt/chrono.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/progress.h>
#include <occ/core/util.h>
#include <occ/interaction/interaction_json.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pair_potential.h>
#include <occ/interaction/pairinteraction.h>

namespace occ::interaction {

using occ::qm::Wavefunction;

PairEnergy::PairEnergy(const occ::io::OccInput &input) {
  if (input.pair.source_a == "none" || input.pair.source_b == "none") {
    throw "Both monomers in a pair energy need a wavefunction set";
  }

  a.wfn = Wavefunction::load(input.pair.source_a);
  b.wfn = Wavefunction::load(input.pair.source_b);
  a.rotation = input.pair.rotation_a;
  b.rotation = input.pair.rotation_b;
  a.translation = input.pair.translation_a * occ::units::ANGSTROM_TO_BOHR;
  b.translation = input.pair.translation_b * occ::units::ANGSTROM_TO_BOHR;

  a.wfn.apply_transformation(a.rotation, a.translation);
  b.wfn.apply_transformation(b.rotation, b.translation);
  occ::log::debug(
      "Transformed atomic positions for monomer A (charge = {}, ecp = {})",
      a.wfn.charge(), a.wfn.basis.ecp_electrons().size() > 0);
  for (const auto &a : a.wfn.atoms) {
    occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}",
                    occ::core::Element(a.atomic_number).symbol(),
                    a.x / occ::units::ANGSTROM_TO_BOHR,
                    a.y / occ::units::ANGSTROM_TO_BOHR,
                    a.z / occ::units::ANGSTROM_TO_BOHR);
  }
  occ::log::debug(
      "Transformed atomic positions for monomer B (charge = {}, ecp = {})",
      b.wfn.charge(), b.wfn.basis.ecp_electrons().size() > 0);
  for (const auto &a : b.wfn.atoms) {
    occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}",
                    occ::core::Element(a.atomic_number).symbol(),
                    a.x / occ::units::ANGSTROM_TO_BOHR,
                    a.y / occ::units::ANGSTROM_TO_BOHR,
                    a.z / occ::units::ANGSTROM_TO_BOHR);
  }
  model = occ::interaction::ce_model_from_string(input.pair.model_name);
}

CEEnergyComponents PairEnergy::compute() {

  CEModelInteraction interaction(model);
  energy = interaction(a.wfn, b.wfn);

  occ::log::info("Monomer A energies\n{}", a.wfn.energy.to_string());
  occ::log::info("Monomer B energies\n{}", b.wfn.energy.to_string());

  occ::log::info("Dimer");

  occ::log::info("Component              Energy (kJ/mol)\n");
  occ::log::info("Coulomb               {: 12.6f}", energy.coulomb_kjmol());
  occ::log::info("Exchange              {: 12.6f}", energy.exchange_kjmol());
  occ::log::info("Repulsion             {: 12.6f}", energy.repulsion_kjmol());
  occ::log::info("Polarization          {: 12.6f}",
                 energy.polarization_kjmol());
  occ::log::info("Dispersion            {: 12.6f}", energy.dispersion_kjmol());
  occ::log::info("__________________________________");
  occ::log::info("Total 		      {: 12.6f}", energy.total_kjmol());
  return energy;
}

} // namespace occ::interaction
