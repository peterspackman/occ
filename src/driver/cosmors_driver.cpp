#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/solvent/cosmors_io.h>
#include <occ/driver/cosmors_driver.h>
#include <occ/qm/scf.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/solvation_correction.h>

namespace occ::driver {

namespace {

void atom_arrays(const std::vector<occ::core::Atom> &atoms, Mat3N &positions,
                 IVec &atomic_numbers) {
  positions.resize(3, atoms.size());
  atomic_numbers.resize(atoms.size());
  for (size_t i = 0; i < atoms.size(); i++) {
    positions(0, i) = atoms[i].x;
    positions(1, i) = atoms[i].y;
    positions(2, i) = atoms[i].z;
    atomic_numbers(i) = atoms[i].atomic_number;
  }
}

} // namespace

solvent::cosmors::Segments
conductor_segments(const qm::Wavefunction &wavefunction,
                   const solvent::cosmors::Parameters &params,
                   double probe_radius_angs, int angular_points,
                   bool constrain_charge, Vec *dielectric_energies,
                   double *cavity_volume_angs3) {
  Mat3N positions;
  IVec atomic_numbers;
  atom_arrays(wavefunction.atoms, positions, atomic_numbers);

  occ::scrf::ReactionFieldEngine engine(
      occ::scrf::Options::conductor(probe_radius_angs, angular_points));
  engine.initialize(positions, atomic_numbers);

  const Vec phi = wavefunction.electric_potential(engine.es_cavity().vertices);
  std::optional<double> constraint;
  if (constrain_charge)
    constraint = static_cast<double>(wavefunction.charge());
  engine.solve_asc(phi, constraint);

  if (dielectric_energies) {
    auto surfaces = engine.surfaces();
    *dielectric_energies = surfaces.coulomb ? surfaces.coulomb->energies
                                            : Vec::Zero(phi.size());
  }

  if (cavity_volume_angs3) {
    constexpr double bohr3_to_angs3 = occ::units::BOHR_TO_ANGSTROM *
                                      occ::units::BOHR_TO_ANGSTROM *
                                      occ::units::BOHR_TO_ANGSTROM;
    *cavity_volume_angs3 =
        occ::solvent::surface::cavity_volume(engine.es_cavity(), positions) *
        bohr3_to_angs3;
  }

  auto segments = solvent::cosmors::segments_from_cavity(
      engine.es_cavity(), engine.surface_charges(), atomic_numbers);
  solvent::cosmors::average_sigma(segments, params.r_av);
  solvent::cosmors::average_sigma_orth(segments, params.r_av, params.r_corr,
                                     params.sigma_orth_factor);
  return segments;
}

ConductorResult conductor_profile(const qm::Wavefunction &gas_wavefunction,
                                  const ConductorSettings &settings) {
  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(gas_wavefunction.atoms, settings.basis);
  basis.set_pure(settings.pure_spherical);

  occ::dft::DFT ks(settings.method, basis);
  occ::solvent::SolvationCorrectedProcedure<occ::dft::DFT> proc(
      ks, occ::scrf::Options::conductor(settings.probe_radius_angs));
  occ::qm::SCF<occ::solvent::SolvationCorrectedProcedure<occ::dft::DFT>> scf(
      proc, gas_wavefunction.mo.kind);
  scf.set_charge_multiplicity(gas_wavefunction.charge(),
                              gas_wavefunction.multiplicity());

  ConductorResult result;
  result.energy_gas = gas_wavefunction.energy.total;
  result.energy_conductor = scf.compute_scf_energy();
  result.wavefunction = scf.wavefunction();
  result.segments = conductor_segments(
      result.wavefunction, settings.parameters, settings.probe_radius_angs,
      settings.angular_points, settings.constrain_charge,
      &result.dielectric_energies, &result.cavity_volume);

  result.cavity_area = result.segments.total_area();
  result.screening_charge = result.segments.total_charge();

  occ::log::info("conductor COSMO: {} segments, area {:.2f} A^2, volume "
                 "{:.2f} A^3, screening charge {:+.4f} e (solute {:+.1f} e)",
                 result.segments.size(), result.cavity_area,
                 result.cavity_volume, result.screening_charge,
                 static_cast<double>(gas_wavefunction.charge()));
  return result;
}

namespace {

ConductorSettings conductor_settings(const CosmoRSSolvationSettings &s) {
  ConductorSettings out;
  out.method = s.method;
  out.basis = s.basis;
  out.pure_spherical = s.pure_spherical;
  out.probe_radius_angs = s.probe_radius_angs;
  out.angular_points = s.angular_points;
  out.constrain_charge = s.constrain_charge;
  return out;
}

/// Gas SCF then conductor SCF for one molecule.
ConductorResult conductor_for(const core::Molecule &molecule,
                              const ConductorSettings &settings) {
  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(molecule.atoms(), settings.basis);
  basis.set_pure(settings.pure_spherical);
  occ::dft::DFT gas(settings.method, basis);
  occ::qm::SCF<occ::dft::DFT> scf(gas);
  scf.compute_scf_energy();
  return conductor_profile(scf.wavefunction(), settings);
}

CosmoRSSolvation assemble(const core::Molecule &solute,
                          const ConductorResult &conductor,
                          const solvent::cosmors::Component &solvent,
                          const CosmoRSSolvationSettings &settings,
                          const solvent::cosmors::Parameters &params) {
  solvent::cosmors::ActivityOptions options;
  options.temperature = settings.temperature;
  const solvent::cosmors::SolventModel model(solvent, params, options);

  const auto component = solvent::cosmors::Component::from_segments(
      conductor.segments, conductor.cavity_volume, conductor.cavity_area);
  const auto missing =
      solvent::cosmors::unparameterised_elements(component, params);
  if (!missing.empty())
    occ::log::warn("{} element(s) have no openCOSMO-RS surface tension and "
                   "contribute nothing to the van der Waals term",
                   missing.size());

  CosmoRSSolvation out;
  out.num_rings = (settings.num_rings >= 0)
                      ? settings.num_rings
                      : solvent::cosmors::ring_count(solute);
  out.energy = solvent::cosmors::solvation_free_energy(
      model, component, conductor.energy_conductor - conductor.energy_gas,
      out.num_rings, settings.liquid_volume);
  out.cavity_area = conductor.cavity_area;
  out.cavity_volume = conductor.cavity_volume;
  out.conductor = conductor.wavefunction;
  return out;
}

} // namespace

CosmoRSSolvation
cosmors_solvation_free_energy(const core::Molecule &solute,
                              const std::string &solvent_name,
                              const CosmoRSSolvationSettings &settings) {
  const auto params = solvent::cosmors::Parameters::v24a();
  auto conductor = conductor_for(solute, conductor_settings(settings));
  const auto solvent = solvent::cosmors::load_solvent(
      solvent::cosmors::SegmentStore::standard(), solvent_name, params,
      settings.method, settings.basis);
  return assemble(solute, conductor, solvent.component, settings, params);
}

CosmoRSSolvation
cosmors_solvation_free_energy(const core::Molecule &solute,
                              const core::Molecule &solvent,
                              const CosmoRSSolvationSettings &settings) {
  const auto params = solvent::cosmors::Parameters::v24a();
  const auto shared = conductor_settings(settings);
  auto solute_conductor = conductor_for(solute, shared);
  auto solvent_conductor = conductor_for(solvent, shared);
  const auto ensemble = solvent::cosmors::Component::from_segments(
      solvent_conductor.segments, solvent_conductor.cavity_volume,
      solvent_conductor.cavity_area);
  return assemble(solute, solute_conductor, ensemble, settings, params);
}

std::vector<std::string> available_cosmors_solvents() {
  return solvent::cosmors::SegmentStore::standard().available();
}

} // namespace occ::driver
