#include <fmt/os.h>
#include <occ/cg/cg_json.h>
#include <occ/cg/smd_solvation.h>
#include <occ/core/log.h>
#include <occ/core/point_group.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>

namespace occ::cg {

SMDCalculator::SMDCalculator(
    const std::string &basename,
    const std::vector<occ::core::Molecule> &molecules,
    const std::vector<occ::qm::Wavefunction> &wavefunctions,
    const std::string &solvent, const SMDSettings &settings)
    : m_basename(basename), m_solvent(solvent), m_settings(settings),
      m_molecules(molecules), m_gas_wavefunctions(wavefunctions) {}

bool SMDCalculator::try_load_cached(const CacheFiles &cache,
                                    SMDSolventSurfaces &surfaces,
                                    occ::qm::Wavefunction &wfn) const {
  if (!cache.exists())
    return false;

  occ::log::info("Loading cached surface properties from {}",
                 cache.surface_path.string());

  std::ifstream ifs(cache.surface_path.string());
  auto jf = nlohmann::json::parse(ifs);
  surfaces = jf.get<SMDSolventSurfaces>();

  occ::log::info("Loading cached solvated wavefunction from {}",
                 cache.wavefunction_path.string());
  wfn = occ::qm::Wavefunction::load(cache.wavefunction_path.string());
  return true;
}

std::pair<SMDSolventSurfaces, occ::qm::Wavefunction>
SMDCalculator::perform_calculation(const occ::core::Molecule &mol,
                                   const occ::qm::Wavefunction &gas_wfn,
                                   size_t index) {
  occ::qm::AOBasis basis =
      occ::qm::AOBasis::load(gas_wfn.atoms, m_settings.basis);
  double original_energy = gas_wfn.energy.total;
  occ::log::debug("Total energy (gas) {:.3f}", original_energy);

  basis.set_pure(m_settings.pure_spherical);
  occ::log::debug("Loaded basis set, {} shells, {} basis functions",
                  basis.size(), basis.nbf());

  // Set up DFT calculation with solvation
  occ::dft::DFT ks(m_settings.method, basis);
  occ::solvent::SolvationCorrectedProcedure<occ::dft::DFT> proc_solv(ks,
                                                                     m_solvent);
  occ::qm::SCF<occ::solvent::SolvationCorrectedProcedure<occ::dft::DFT>> scf(
      proc_solv, gas_wfn.mo.kind);

  scf.convergence_settings.incremental_fock_threshold =
      m_settings.convergence_threshold;
  scf.set_charge_multiplicity(gas_wfn.charge(), gas_wfn.multiplicity());

  // Perform SCF calculation
  double solvated_energy = scf.compute_scf_energy();
  auto solvated_wfn = scf.wavefunction();

  // Collect surface data
  SMDSolventSurfaces surfaces;
  surfaces.coulomb.positions = proc_solv.surface_positions_coulomb();
  surfaces.cds.positions = proc_solv.surface_positions_cds();
  surfaces.coulomb.areas = proc_solv.surface_areas_coulomb();
  surfaces.cds.areas = proc_solv.surface_areas_cds();
  surfaces.cds.energies = proc_solv.surface_cds_energy_elements();

  auto nuc = proc_solv.surface_nuclear_energy_elements();
  auto elec = proc_solv.surface_electronic_energy_elements(scf.ctx.mo);
  auto pol = proc_solv.surface_polarization_energy_elements();
  surfaces.coulomb.energies = nuc + elec + pol;

  // Debug energy components
  occ::log::debug("sum e_nuc {:12.6f}", nuc.array().sum());
  occ::log::debug("sum e_ele {:12.6f}", elec.array().sum());
  occ::log::debug("sum e_pol {:12.6f}", pol.array().sum());
  occ::log::debug("sum e_cds {:12.6f}", surfaces.cds.energies.array().sum());

  double surface_energy = nuc.array().sum() + elec.array().sum() +
                          pol.array().sum() +
                          surfaces.cds.energies.array().sum();

  calculate_free_energy_components(surfaces, mol, original_energy,
                                   solvated_energy, surface_energy);

  surfaces.electronic_energies = (surfaces.electronic_contribution /
                                  surfaces.coulomb.areas.array().sum()) *
                                 surfaces.coulomb.areas.array();

  return {surfaces, solvated_wfn};
}

void SMDCalculator::save_calculation(const CacheFiles &cache,
                                     const SMDSolventSurfaces &surfaces,
                                     occ::qm::Wavefunction &wfn) const {
  occ::log::info("Writing solvated surface properties to {}",
                 cache.surface_path.string());
  {
    std::ofstream ofs(cache.surface_path.string());
    nlohmann::json j = surfaces;
    ofs << j;
  }

  occ::log::info("Writing solvated wavefunction to {}",
                 cache.wavefunction_path.string());
  wfn.save(cache.wavefunction_path.string());
}

void SMDCalculator::calculate_free_energy_components(
    SMDSolventSurfaces &surfaces, const occ::core::Molecule &mol,
    double original_energy, double solvated_energy,
    double surface_energy) const {

  // Calculate rotational and translational contributions
  double Gr = mol.rotational_free_energy(m_settings.temperature);
  occ::core::MolecularPointGroup pg(mol);
  double Gt = mol.translational_free_energy(m_settings.temperature);

  // Temperature-dependent terms
  const double R = 8.31446261815324;
  const double RT = m_settings.temperature * R / 1000;
  Gr += RT * std::log(pg.symmetry_number());

  // Set free energy components
  surfaces.free_energy_correction =
      (1.89 / occ::units::KJ_TO_KCAL - 2 * RT) / occ::units::AU_TO_KJ_PER_MOL;
  surfaces.gas_phase_contribution = (Gt + Gr) / occ::units::AU_TO_KJ_PER_MOL;
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
    CacheFiles cache(m_basename, i, m_solvent);

    SMDSolventSurfaces surfaces;
    occ::qm::Wavefunction wavefunction;

    bool cached = try_load_cached(cache, surfaces, wavefunction);
    if (!cached) {
      std::tie(surfaces, wavefunction) =
          perform_calculation(m_molecules[i], m_gas_wavefunctions[i], i);
      save_calculation(cache, surfaces, wavefunction);
    }
    result.surfaces.push_back(surfaces);
    result.wavefunctions.push_back(wavefunction);
  }

  return result;
}

} // namespace occ::cg
