#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/dimer_desolvation.h>
#include <occ/driver/sigma_solvation.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/sigma_solvation.h>

namespace occ::driver {

namespace {

/// Move a monomer wavefunction onto the position of `mol` in the crystal.
qm::Wavefunction placed(const qm::Wavefunction &wfn, const core::Molecule &mol,
                        const crystal::Crystal &crystal) {
  auto transform =
      occ::interaction::transform::WavefunctionTransformer::calculate_transform(
          wfn, mol, crystal);
  qm::Wavefunction out = wfn;
  out.apply_transformation(transform.rotation, transform.translation);
  return out;
}

/// Conductor solvation of a set of molecules whose potential is the
/// superposition of the given wavefunctions. Returns E_diel + residual, in
/// Hartree.
double conductor_solvation(const std::vector<qm::Wavefunction> &wavefunctions,
                           const solvent::sigma::SolventModel &solvent,
                           const solvent::sigma::Parameters &params,
                           int angular_points) {
  std::vector<core::Atom> atoms;
  for (const auto &wfn : wavefunctions)
    atoms.insert(atoms.end(), wfn.atoms.begin(), wfn.atoms.end());

  const Eigen::Index natoms = atoms.size();
  Mat3N positions(3, natoms);
  IVec atomic_numbers(natoms);
  for (Eigen::Index i = 0; i < natoms; i++) {
    positions(0, i) = atoms[i].x;
    positions(1, i) = atoms[i].y;
    positions(2, i) = atoms[i].z;
    atomic_numbers(i) = atoms[i].atomic_number;
  }

  occ::scrf::ReactionFieldEngine engine(
      occ::scrf::Options::conductor(0.0, angular_points));
  engine.initialize(positions, atomic_numbers);

  const auto &cavity = engine.es_cavity();
  Vec phi = Vec::Zero(cavity.areas.size());
  for (const auto &wfn : wavefunctions)
    phi += wfn.electric_potential(cavity.vertices);

  double total_charge = 0.0;
  for (const auto &wfn : wavefunctions)
    total_charge += wfn.charge();
  engine.solve_asc(phi, total_charge);

  // E_diel is the variational screening energy on this cavity.
  auto surfaces = engine.surfaces();
  const double e_diel =
      surfaces.coulomb ? surfaces.coulomb->energies.sum() : 0.0;

  auto segments = solvent::sigma::segments_from_cavity(
      cavity, engine.surface_charges(), atomic_numbers);
  solvent::sigma::classify_hbond_segments(segments, atomic_numbers, positions);
  solvent::sigma::average_sigma(segments, params.r_av, params.f_decay);

  return e_diel + solvent.segment_energies(segments).sum();
}

} // namespace

std::vector<double> dimer_desolvation(
    const crystal::Crystal &crystal,
    const std::vector<qm::Wavefunction> &conductor_wavefunctions,
    const crystal::CrystalDimers &dimers, const SolventSpec &solvent,
    const SigmaSolvationSettings &settings, double max_distance) {

  const auto params = solvent::sigma::Parameters::for_model(settings.model);
  solvent::sigma::PotentialOptions options;
  options.temperature = settings.temperature;

  auto store = solvent::sigma::ProfileStore::standard();
  std::vector<solvent::sigma::Component> components;
  for (const auto &name : solvent.components)
    components.push_back(store.get(name));
  auto mixture = solvent.is_mixture()
                     ? solvent::sigma::mix_components(components,
                                                      solvent.mole_fractions)
                     : components.front();
  solvent::sigma::SolventModel model(std::move(mixture), params, options);

  // Monomer solvation, once per symmetry-unique molecule.
  const auto molecules = crystal.symmetry_unique_molecules();
  std::vector<double> monomer(molecules.size(), 0.0);
  for (size_t i = 0; i < molecules.size(); i++) {
    auto wfn = placed(conductor_wavefunctions[i], molecules[i], crystal);
    monomer[i] = conductor_solvation({wfn}, model, params,
                                     settings.angular_points);
  }

  std::vector<double> result(dimers.unique_dimers.size(), 0.0);
  size_t computed = 0;
  for (size_t k = 0; k < dimers.unique_dimers.size(); k++) {
    const auto &dimer = dimers.unique_dimers[k];
    // Distant dimers bury no surface, so they get no attributed solvation.
    if (dimer.nearest_distance() > max_distance)
      continue;
    computed++;
    const auto a = dimer.a();
    const auto b = dimer.b();
    const int idx_a = a.asymmetric_molecule_idx();
    const int idx_b = b.asymmetric_molecule_idx();

    auto wfn_a = placed(conductor_wavefunctions[idx_a], a, crystal);
    auto wfn_b = placed(conductor_wavefunctions[idx_b], b, crystal);
    const double together =
        conductor_solvation({wfn_a, wfn_b}, model, params,
                            settings.angular_points);

    // G_solv(AB) - G_solv(A) - G_solv(B): positive when forming the contact
    // costs solvation, since the buried region is no longer solvated.
    result[k] = together - monomer[idx_a] - monomer[idx_b];
    occ::log::debug("dimer {}: desolvation {:.4f} kJ/mol", k,
                    result[k] * occ::units::AU_TO_KJ_PER_MOL);
  }
  occ::log::info("Dimer-difference desolvation: {} of {} unique dimers within "
                 "{:.1f} Angstrom",
                 computed, dimers.unique_dimers.size(), max_distance);
  return result;
}

} // namespace occ::driver
