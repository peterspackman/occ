#include <fmt/os.h>
#include <occ/cg/cg_json.h>
#include <occ/cg/smd_solvation.h>
#include <occ/cg/solvation_data.h>
#include <occ/core/point_group.h>
#include <occ/driver/solvated_procedure.h>
#include <occ/qm/io/wavefunction_json.h>
#include <occ/qm/scf.h>
#include <tuple>
#include <type_traits>

namespace occ::cg {

SMDCalculator::SMDCalculator(
    const std::string &basename,
    const std::vector<occ::core::Molecule> &molecules,
    const std::vector<occ::qm::Wavefunction> &wavefunctions,
    const std::string &solvent, occ::io::JsonCache &cache,
    const SMDSettings &settings)
    : m_basename(basename), m_solvent(solvent), m_settings(settings),
      m_molecules(molecules), m_gas_wavefunctions(wavefunctions),
      m_cache(cache) {}

bool SMDCalculator::try_load_cached(const CacheKeys &keys,
                                    SolvationData &surfaces,
                                    occ::qm::Wavefunction &wfn) const {
  const auto wfn_doc = m_cache.load(keys.wavefunction);
  if (!wfn_doc)
    return false;
  const auto surface_doc = m_cache.load(keys.surface);
  if (!surface_doc)
    return false;

  // Both documents depend on the level of theory, which the keys do not
  // record: reuse them only when the cached wavefunction was computed at this
  // one. A wavefunction saved without its method says "SCF" and is judged on
  // the basis alone.
  auto cached_wfn = wfn_doc->get<occ::qm::Wavefunction>();
  const bool same_level =
      cached_wfn.basis.name() == m_settings.basis &&
      cached_wfn.basis.is_pure() == m_settings.pure_spherical &&
      (cached_wfn.method == "SCF" || cached_wfn.method == m_settings.method);
  if (!same_level) {
    occ::log::warn("Cached solvated wavefunction in {} was computed at a "
                   "different level ({}/{}); recomputing at {}/{}",
                   m_cache.location(keys.wavefunction), cached_wfn.method,
                   cached_wfn.basis.name(), m_settings.method,
                   m_settings.basis);
    return false;
  }

  occ::log::info("Loading cached surface properties from {}",
                 m_cache.location(keys.surface));
  surfaces = surface_doc->get<SolvationData>();

  occ::log::info("Loading cached solvated wavefunction from {}",
                 m_cache.location(keys.wavefunction));
  wfn = std::move(cached_wfn);
  return true;
}

std::pair<SolvationData, occ::qm::Wavefunction>
SMDCalculator::perform_calculation(const occ::core::Molecule &mol,
                                   const occ::qm::Wavefunction &gas_wfn,
                                   size_t index) {
  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(gas_wfn.atoms, m_settings.basis);
  double original_energy = gas_wfn.energy.total;
  occ::log::debug("Total energy (gas) {:.3f}", original_energy);

  basis.set_pure(m_settings.pure_spherical);
  occ::log::debug("Loaded basis set, {} shells, {} basis functions",
                  basis.size(), basis.nbf());

  // The method comes from the energy model, so it may be HF (CE-HF) or DFT.
  auto [solvated_energy, solvated_wfn, scrf_surfaces] =
      occ::driver::with_solvated_procedure(
          m_settings.method, basis,
          [&](auto &proc_solv) {
            occ::qm::SCF<std::remove_reference_t<decltype(proc_solv)>> scf(
                proc_solv, gas_wfn.mo.kind);
            scf.set_charge_multiplicity(gas_wfn.charge(),
                                        gas_wfn.multiplicity());
            const double energy = scf.compute_scf_energy();
            // The SCRF engine reports per-element ES energies as
            // ½σ_i·φ_total_i, which is algebraically identical to the legacy
            // `nuc_i + elec_i + pol_i` decomposition (see
            // `from_scrf_surfaces`); summed totals match to floating-point
            // precision.
            return std::make_tuple(energy, scf.wavefunction(),
                                   proc_solv.solvation_surfaces());
          },
          m_solvent);

  SolvationData surfaces = occ::cg::from_scrf_surfaces(scrf_surfaces);

  double surface_energy = surfaces.total_energy();
  occ::log::debug("sum e_surface {:12.6f}", surface_energy);

  calculate_free_energy_components(surfaces, mol, original_energy,
                                   solvated_energy, surface_energy);

  // The electronic relaxation has no per-element decomposition, so it is
  // spread by area over the electrostatic cavity as its own channel.
  if (auto *cavity = occ::cg::coulomb_cavity(surfaces)) {
    cavity->energies.push_back(
        {"electronic",
         (surfaces.electronic_contribution / cavity->areas.array().sum()) *
             cavity->areas.array()});
  }

  return {surfaces, solvated_wfn};
}

void SMDCalculator::save_calculation(const CacheKeys &keys,
                                     const SolvationData &surfaces,
                                     occ::qm::Wavefunction &wfn) const {
  occ::log::info("Caching solvated surface properties in {}",
                 m_cache.location(keys.surface));
  m_cache.store(keys.surface, surfaces);

  occ::log::info("Caching solvated wavefunction in {}",
                 m_cache.location(keys.wavefunction));
  wfn.method = m_settings.method; // recorded for cache validation
  m_cache.store(keys.wavefunction, wfn);
}

void SMDCalculator::calculate_free_energy_components(
    SolvationData &surfaces, const occ::core::Molecule &mol,
    double original_energy, double solvated_energy,
    double surface_energy) const {

  surfaces.electronic_contribution =
      solvated_energy - original_energy - surface_energy;
  surfaces.total_solvation_energy = solvated_energy - original_energy;

  // Log energetics
  occ::log::debug("total e_solv {:12.6f} ({:.3f} kJ/mol)", surface_energy,
                  surface_energy * occ::units::AU_TO_KJ_PER_MOL);
  occ::log::info("SCF difference         (au)       {: 9.3f}",
                 solvated_energy - original_energy);
  occ::log::debug("SCF difference         (kJ/mol)   {: 9.3f}",
                  occ::units::AU_TO_KJ_PER_MOL *
                      (solvated_energy - original_energy));
  occ::log::debug("total E solv (surface) (kj/mol)   {: 9.3f}",
                  surface_energy * occ::units::AU_TO_KJ_PER_MOL);
}

SMDCalculator::Result SMDCalculator::calculate() {
  Result result;
  result.surfaces.reserve(m_gas_wavefunctions.size());
  result.wavefunctions.reserve(m_gas_wavefunctions.size());

  for (size_t i = 0; i < m_gas_wavefunctions.size(); ++i) {
    const CacheKeys keys(m_basename, i, m_solvent);

    SolvationData surfaces;
    occ::qm::Wavefunction wavefunction;

    bool cached = try_load_cached(keys, surfaces, wavefunction);
    if (!cached) {
      std::tie(surfaces, wavefunction) =
          perform_calculation(m_molecules[i], m_gas_wavefunctions[i], i);
      save_calculation(keys, surfaces, wavefunction);
    }
    result.surfaces.push_back(surfaces);
    result.wavefunctions.push_back(wavefunction);
  }

  return result;
}

} // namespace occ::cg
