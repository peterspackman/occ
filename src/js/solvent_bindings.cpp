#include "solvent_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/units.h>
#include <occ/driver/cosmors_driver.h>

using namespace emscripten;
using occ::driver::CosmoRSSolvation;
using occ::driver::CosmoRSSolvationSettings;

namespace {

/// The term breakdown as a plain JS object, in kJ/mol. Hartree is the C++
/// unit, but nothing on the JS side works in atomic units.
val energy_to_val(const occ::solvent::cosmors::SolvationEnergy &e) {
  const double k = occ::units::AU_TO_KJ_PER_MOL;
  val o = val::object();
  o.set("dielectric", e.dielectric * k);
  o.set("residual", e.residual * k);
  o.set("combinatorial", e.combinatorial * k);
  o.set("vdw", e.vdw * k);
  o.set("ring", e.ring * k);
  o.set("referenceState", e.reference_state * k);
  o.set("eta", e.eta * k);
  o.set("total", e.total() * k);
  return o;
}

val solvation_to_val(const CosmoRSSolvation &r) {
  val o = val::object();
  o.set("energy", energy_to_val(r.energy));
  o.set("cavityArea", r.cavity_area);
  o.set("cavityVolume", r.cavity_volume);
  o.set("numRings", r.num_rings);
  o.set("total", r.total() * occ::units::AU_TO_KJ_PER_MOL);
  return o;
}

val solvation_free_energy(const occ::core::Molecule &solute,
                          const std::string &solvent,
                          const CosmoRSSolvationSettings &settings) {
  return solvation_to_val(
      occ::driver::cosmors_solvation_free_energy(solute, solvent, settings));
}

val solvation_free_energy_in_molecule(
    const occ::core::Molecule &solute, const occ::core::Molecule &solvent,
    const CosmoRSSolvationSettings &settings) {
  return solvation_to_val(
      occ::driver::cosmors_solvation_free_energy(solute, solvent, settings));
}

} // namespace

void register_solvent_bindings() {

  class_<CosmoRSSolvationSettings>("CosmoRSSettings")
      .constructor<>()
      .property("method", &CosmoRSSolvationSettings::method)
      .property("basis", &CosmoRSSolvationSettings::basis)
      .property("pureSpherical", &CosmoRSSolvationSettings::pure_spherical)
      .property("probeRadius", &CosmoRSSolvationSettings::probe_radius_angs)
      .property("angularPoints", &CosmoRSSolvationSettings::angular_points)
      .property("constrainCharge", &CosmoRSSolvationSettings::constrain_charge)
      .property("temperature", &CosmoRSSolvationSettings::temperature)
      .property("liquidVolume", &CosmoRSSolvationSettings::liquid_volume)
      .property("numRings", &CosmoRSSolvationSettings::num_rings);

  // Both return a plain JS object:
  // {energy: {...}, cavityArea, cavityVolume, numRings, total} in kJ/mol.
  function("cosmoRsSolvationFreeEnergy", &solvation_free_energy);
  function("cosmoRsSolvationFreeEnergyInSolventGeometry",
           &solvation_free_energy_in_molecule);
  function("availableCosmoRsSolvents",
           &occ::driver::available_cosmors_solvents);
}
