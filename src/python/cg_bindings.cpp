#include "cg_bindings.h"
#include <fmt/core.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/interaction/pair_energy.h>
#include <occ/main/occ_cg.h>

using namespace nb::literals;
using occ::cg::CrystalGrowthResult;
using occ::cg::DimerResult;
using occ::cg::DimerSolventTerm;
using occ::cg::MoleculeResult;
using occ::interaction::LatticeConvergenceSettings;
using occ::main::CGConfig;

template <typename K, typename V>
inline nb::dict
convert_map_to_dict(const ankerl::unordered_dense::map<K, V> &map) {
  nb::dict result;
  for (const auto &[key, value] : map) {
    result[nb::cast(key)] = nb::cast(value);
  }
  return result;
}

nb::module_ register_cg_bindings(nb::module_ &m) {

  nb::class_<LatticeConvergenceSettings>(m, "LatticeConvergenceSettings")
      .def(nb::init<>())
      .def_rw("min_radius", &LatticeConvergenceSettings::min_radius)
      .def_rw("max_radius", &LatticeConvergenceSettings::max_radius)
      .def_rw("radius_increment", &LatticeConvergenceSettings::radius_increment)
      .def_rw("energy_tolerance", &LatticeConvergenceSettings::energy_tolerance)
      .def_rw("wolf_sum", &LatticeConvergenceSettings::wolf_sum)
      .def_rw("crystal_field_polarization",
              &LatticeConvergenceSettings::crystal_field_polarization)
      .def_rw("model_name", &LatticeConvergenceSettings::model_name)
      .def_rw("crystal_filename", &LatticeConvergenceSettings::crystal_filename)
      .def_rw("output_json_filename",
              &LatticeConvergenceSettings::output_json_filename);

  nb::class_<CGConfig>(m, "CrystalGrowthConfig")
      .def(nb::init<>())
      .def_rw("lattice_settings", &CGConfig::lattice_settings)
      .def_rw("cg_radius", &CGConfig::cg_radius)
      .def_rw("solvent", &CGConfig::solvent)
      .def_rw("wavefunction_choice", &CGConfig::wavefunction_choice)
      .def_rw("num_surface_energies", &CGConfig::max_facets);

  nb::class_<DimerSolventTerm>(m, "DimerSolventTerm")
      .def_ro("ab", &DimerSolventTerm::ab)
      .def_ro("ba", &DimerSolventTerm::ba)
      .def_ro("total", &DimerSolventTerm::total);

  nb::class_<DimerResult>(m, "DimerResult")
      .def_ro("dimer", &DimerResult::dimer)
      .def_ro("unique_idx", &DimerResult::unique_idx)
      .def("total_energy", &DimerResult::total_energy)
      .def("energy_component", &DimerResult::energy_component)
      .def("energy_components",
           [](const DimerResult &d) {
             return convert_map_to_dict(d.energy_components);
           })
      .def_ro("is_nearest_neighbor", &DimerResult::is_nearest_neighbor);

  nb::class_<MoleculeResult>(m, "MoleculeResult")
      .def_ro("dimer_results", &MoleculeResult::dimer_results)
      .def_ro("total", &MoleculeResult::total)
      .def_ro("has_inversion_symmetry", &MoleculeResult::has_inversion_symmetry)
      .def("total_energy", &MoleculeResult::total_energy)
      .def("energy_components",
           [](const MoleculeResult &m) {
             return convert_map_to_dict(m.energy_components);
           })
      .def("energy_component", &MoleculeResult::energy_component);

  nb::class_<CrystalGrowthResult>(m, "CrystalGrowthResult")
      .def_ro("molecule_results", &CrystalGrowthResult::molecule_results);

  nb::class_<occ::cg::EnergyTotal>(m, "CrystalGrowthEnergyTotal")
      .def_ro("crystal", &occ::cg::EnergyTotal::crystal_energy)
      .def_ro("int", &occ::cg::EnergyTotal::interaction_energy)
      .def_ro("solution", &occ::cg::EnergyTotal::solution_term)
      .def("__repr__", [](const occ::cg::EnergyTotal &tot) {
        return fmt::format("(crys={:.6f}, int={:.6f}, sol={:.6f})",
                           tot.crystal_energy, tot.interaction_energy,
                           tot.solution_term);
      });

  return m;
}
