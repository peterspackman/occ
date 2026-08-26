#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/driver/sigma_driver.h>
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

solvent::sigma::Segments
conductor_segments(const qm::Wavefunction &wavefunction,
                   const solvent::sigma::Parameters &params,
                   double probe_radius_angs, int angular_points,
                   bool constrain_charge) {
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

  auto segments = solvent::sigma::segments_from_cavity(
      engine.es_cavity(), engine.surface_charges(), atomic_numbers);
  solvent::sigma::classify_hbond_segments(segments, atomic_numbers, positions);
  solvent::sigma::average_sigma(segments, params.r_av, params.f_decay);
  return segments;
}

ConductorResult conductor_profile(const qm::Wavefunction &gas_wavefunction,
                                  const SigmaProfileSettings &settings) {
  const auto params = solvent::sigma::Parameters::for_model(settings.model);

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
  result.segments =
      conductor_segments(result.wavefunction, params,
                         settings.probe_radius_angs, settings.angular_points,
                         settings.constrain_charge);

  Mat3N positions;
  IVec atomic_numbers;
  atom_arrays(result.wavefunction.atoms, positions, atomic_numbers);

  occ::scrf::ReactionFieldEngine engine(
      occ::scrf::Options::conductor(settings.probe_radius_angs,
                                    settings.angular_points));
  engine.initialize(positions, atomic_numbers);
  const double bohr3_to_angs3 = occ::units::BOHR_TO_ANGSTROM *
                                occ::units::BOHR_TO_ANGSTROM *
                                occ::units::BOHR_TO_ANGSTROM;
  result.cavity_area = result.segments.total_area();
  result.cavity_volume =
      occ::solvent::surface::cavity_volume(engine.es_cavity(), positions) *
      bohr3_to_angs3;
  result.screening_charge = result.segments.total_charge();

  occ::log::info("conductor COSMO: {} segments, area {:.2f} A^2, volume "
                 "{:.2f} A^3, screening charge {:+.4f} e (solute {:+.1f} e)",
                 result.segments.size(), result.cavity_area,
                 result.cavity_volume, result.screening_charge,
                 static_cast<double>(gas_wavefunction.charge()));
  return result;
}

} // namespace occ::driver
