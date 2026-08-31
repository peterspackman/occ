#include "cg_bindings.h"
#include <fmt/core.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/cg/interaction_mapper.h>
#include <occ/interaction/pair_energy.h>
#include <occ/driver/cg_runner.h>

using namespace nb::literals;
using occ::cg::CrystalGrowthResult;
using occ::cg::DimerResult;
using occ::cg::DimerSolventTerm;
using occ::cg::InteractionMapper;
using occ::cg::MoleculeResult;
using occ::interaction::LatticeConvergenceSettings;
using occ::driver::CGConfig;
using occ::driver::DMAReferenceLevel;

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

  nb::class_<DMAReferenceLevel>(m, "DMAReferenceLevel")
      .def(nb::init<>())
      .def_rw("model", &DMAReferenceLevel::model)
      .def_rw("method", &DMAReferenceLevel::method)
      .def_rw("basis", &DMAReferenceLevel::basis);

  nb::class_<CGConfig>(m, "CrystalGrowthConfig")
      .def(nb::init<>())
      .def_rw("lattice_settings", &CGConfig::lattice_settings)
      .def_rw("cg_radius", &CGConfig::cg_radius)
      .def_rw("solvent", &CGConfig::solvent)
      .def_rw("solvation_model", &CGConfig::solvation_model,
              "smd (default), cosmo-rs, or none")
      .def_rw("temperature", &CGConfig::temperature,
              "temperature in K for the solvation model")
      .def_rw("charge_string", &CGConfig::charge_string)
      .def_rw("dma_reference", &CGConfig::dma_reference)
      .def_rw("wavefunction_choice", &CGConfig::wavefunction_choice)
      .def_rw("compute_morphology", &CGConfig::compute_morphology)
      .def_rw("num_surface_energies", &CGConfig::max_facets);

  nb::class_<occ::cg::FacetMorphology>(m, "FacetMorphology")
      .def_ro("hkl", &occ::cg::FacetMorphology::hkl)
      .def_ro("gamma", &occ::cg::FacetMorphology::gamma)
      .def_ro("area", &occ::cg::FacetMorphology::area);

  nb::class_<occ::cg::EdgeMorphology>(m, "EdgeMorphology")
      .def_ro("hkl_a", &occ::cg::EdgeMorphology::hkl_a)
      .def_ro("hkl_b", &occ::cg::EdgeMorphology::hkl_b)
      .def_ro("length", &occ::cg::EdgeMorphology::length)
      .def_ro("line_tension", &occ::cg::EdgeMorphology::lambda);

  nb::class_<occ::cg::CornerMorphology>(m, "CornerMorphology")
      .def_ro("hkls", &occ::cg::CornerMorphology::hkls)
      .def_ro("count", &occ::cg::CornerMorphology::count)
      .def_ro("epsilon", &occ::cg::CornerMorphology::epsilon);

  nb::class_<occ::cg::ParticleSample>(m, "ParticleSample")
      .def_ro("size_scale", &occ::cg::ParticleSample::size_scale)
      .def_ro("n_molecules", &occ::cg::ParticleSample::n_molecules)
      .def_ro("e_excess", &occ::cg::ParticleSample::e_excess)
      .def_ro("e_surface", &occ::cg::ParticleSample::e_surface)
      .def_ro("e_edge", &occ::cg::ParticleSample::e_edge)
      .def_ro("e_corner", &occ::cg::ParticleSample::e_corner)
      .def_ro("e_surface_analytic",
              &occ::cg::ParticleSample::e_surface_analytic)
      .def_ro("area", &occ::cg::ParticleSample::area)
      .def_ro("edge_length", &occ::cg::ParticleSample::edge_length)
      .def_ro("n_corners", &occ::cg::ParticleSample::n_corners);

  nb::class_<occ::cg::MorphologyResult>(m, "MorphologyResult")
      .def_ro("shape", &occ::cg::MorphologyResult::shape)
      .def_ro("mu_bulk", &occ::cg::MorphologyResult::mu_bulk)
      .def_ro("molecular_volume", &occ::cg::MorphologyResult::molecular_volume)
      .def_ro("facets", &occ::cg::MorphologyResult::facets)
      .def_ro("edges", &occ::cg::MorphologyResult::edges)
      .def_ro("corners", &occ::cg::MorphologyResult::corners)
      .def_ro("samples", &occ::cg::MorphologyResult::samples)
      .def("empty", &occ::cg::MorphologyResult::empty);

  nb::class_<DimerSolventTerm>(m, "DimerSolventTerm")
      .def_ro("ab", &DimerSolventTerm::ab)
      .def_ro("ba", &DimerSolventTerm::ba)
      .def_ro("total", &DimerSolventTerm::total);

  nb::class_<DimerResult>(m, "DimerResult")
      .def(nb::init<occ::core::Dimer &, bool, int>())
      .def_ro("dimer", &DimerResult::dimer)
      .def_ro("unique_idx", &DimerResult::unique_idx)
      .def("set_energy_component", &DimerResult::set_energy_component)
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
      .def_ro("molecule_results", &CrystalGrowthResult::molecule_results)
      .def_ro("morphology", &CrystalGrowthResult::morphology);

  nb::class_<occ::cg::EnergyTotal>(m, "CrystalGrowthEnergyTotal")
      .def_ro("crystal", &occ::cg::EnergyTotal::crystal_energy)
      .def_ro("interaction", &occ::cg::EnergyTotal::interaction_energy)
      .def_ro("solution", &occ::cg::EnergyTotal::solution_term)
      .def("__repr__", [](const occ::cg::EnergyTotal &tot) {
        return fmt::format("(crys={:.6f}, int={:.6f}, sol={:.6f})",
                           tot.crystal_energy, tot.interaction_energy,
                           tot.solution_term);
      });

  nb::class_<InteractionMapper>(m, "InteractionMapper")
      .def(nb::init<const occ::crystal::Crystal &,
                    const occ::crystal::CrystalDimers &,
                    occ::crystal::CrystalDimers &, bool>())
      .def("map_interactions", &InteractionMapper::map_interactions);

  return m;
}
