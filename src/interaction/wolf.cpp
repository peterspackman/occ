#include <cmath>
#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/wolf.h>
#include <unsupported/Eigen/SpecialFunctions>

namespace occ::interaction {

double wolf_coulomb_energy(double qi, const Vec3 &pi, Eigen::Ref<const Vec> qj,
                           Eigen::Ref<const Mat3N> pj,
                           const WolfParameters &params) {
  using occ::constants::sqrt_pi;
  using std::erfc;
  double eta = params.alpha / occ::units::ANGSTROM_TO_BOHR;
  double rc = params.cutoff * occ::units::ANGSTROM_TO_BOHR;
  double trc = erfc(eta * rc) / rc;

  double self_term = qi * qi * (0.5 * trc + eta / sqrt_pi<double>);
  Vec rij = (pj.colwise() - pi).colwise().norm() * occ::units::ANGSTROM_TO_BOHR;

  double pair_term =
      qi *
      (qj.array() * ((eta * rij).array().erfc() / rij.array() - trc)).sum();
  return 0.5 * pair_term - self_term;
}

double wolf_pair_energy(Eigen::Ref<const Vec> charges_a,
                        Eigen::Ref<const Mat3N> positions_a,
                        Eigen::Ref<const Vec> charges_b,
                        Eigen::Ref<const Mat3N> positions_b,
                        const WolfParameters &params) {
  using occ::constants::sqrt_pi;
  using std::erfc;

  double eta = params.alpha / occ::units::ANGSTROM_TO_BOHR;
  double rc = params.cutoff * occ::units::ANGSTROM_TO_BOHR;
  double trc = erfc(eta * rc) / rc;

  size_t n_a = charges_a.size();
  size_t n_b = charges_b.size();

  double energy = 0.0;

  // Compute pairwise interaction between molecules A and B
  // E = Σ_i∈A Σ_j∈B q_i q_j [erfc(η*r_ij)/r_ij - erfc(η*r_c)/r_c]
  for (size_t i = 0; i < n_a; i++) {
    Vec3 pos_a = positions_a.col(i);
    double q_a = charges_a(i);

    for (size_t j = 0; j < n_b; j++) {
      Vec3 pos_b = positions_b.col(j);
      double q_b = charges_b(j);

      Vec3 r_vec = pos_b - pos_a;
      double r = r_vec.norm();

      if (r < 1e-10) continue; // Skip if somehow same position

      // Wolf interaction: q_i * q_j * [erfc(η*r)/r - t(r_c)]
      double interaction = q_a * q_b * (erfc(eta * r) / r - trc);
      energy += interaction;
    }
  }

  // Energy is in atomic units (Hartree)
  return energy;
}

Mat3N wolf_electric_field(Eigen::Ref<const Vec> charges,
                          Eigen::Ref<const Mat3N> source_positions,
                          Eigen::Ref<const Mat3N> target_positions,
                          const WolfParameters &params) {
  using occ::constants::sqrt_pi;
  using std::erfc;

  // Convert parameters to atomic units (Bohr)
  double eta = params.alpha / occ::units::ANGSTROM_TO_BOHR;
  double rc = params.cutoff * occ::units::ANGSTROM_TO_BOHR;

  size_t n_target = target_positions.cols();
  size_t n_source = source_positions.cols();

  Mat3N electric_field = Mat3N::Zero(3, n_target);

  for (size_t i = 0; i < n_target; i++) {
    Vec3 r_target = target_positions.col(i);

    for (size_t j = 0; j < n_source; j++) {
      Vec3 r_source = source_positions.col(j);
      Vec3 r_vec = r_target - r_source;
      double r = r_vec.norm();

      if (r < 1e-10) continue; // Skip self-interaction

      double q = charges(j);
      double r2 = r * r;
      double r3 = r2 * r;

      // Wolf electric field: E = q * [erfc(η*r)/r^3 + 2η/(√π·r^2)·exp(-η²r²)] * r_vec
      double erfc_term = erfc(eta * r) / r3;
      double exp_term = exp(-eta * eta * r2);
      double gaussian_term = 2.0 * eta / (sqrt_pi<double> * r2) * exp_term;

      double field_magnitude = q * (erfc_term + gaussian_term);

      electric_field.col(i) += field_magnitude * r_vec;
    }
  }

  // Electric field is in atomic units (e/Bohr^2)
  return electric_field;
}

WolfCouplingResult compute_wolf_coupling_terms(
    const std::vector<Mat3N> &electric_fields_per_neighbor,
    Eigen::Ref<const Vec> polarizabilities) {

  WolfCouplingResult result;
  result.total_coupling = 0.0;

  size_t n_neighbors = electric_fields_per_neighbor.size();
  if (n_neighbors < 2) {
    return result; // Need at least 2 neighbors for coupling
  }

  size_t n_atoms = polarizabilities.size();

  // Compute all pairwise coupling terms: C_AB = -Σ_i α_i E_A(i)·E_B(i)
  for (size_t a = 0; a < n_neighbors; a++) {
    const Mat3N &field_a = electric_fields_per_neighbor[a];

    for (size_t b = a + 1; b < n_neighbors; b++) {
      const Mat3N &field_b = electric_fields_per_neighbor[b];

      double coupling = 0.0;

      // Sum over all atoms in the target molecule
      for (size_t i = 0; i < n_atoms; i++) {
        Vec3 E_a = field_a.col(i);
        Vec3 E_b = field_b.col(i);
        double alpha_i = polarizabilities(i);

        // Coupling contribution: α_i * E_A·E_B
        coupling += alpha_i * E_a.dot(E_b);
      }

      // Apply negative sign as per the derivation
      coupling = -coupling;

      result.coupling_terms.push_back({a, b, coupling});
      result.total_coupling += coupling;

      occ::log::debug("Coupling between neighbors {} and {}: {:.8f} Ha", a, b,
                      coupling);
    }
  }

  occ::log::info("Total coupling energy from {} terms: {:.8f} Ha ({:.2f} kJ/mol)",
                 result.coupling_terms.size(), result.total_coupling,
                 result.total_coupling * units::AU_TO_KJ_PER_MOL);

  return result;
}

WolfSum::WolfSum(WolfParameters params) : m_params(params) {}

void WolfSum::initialize(const crystal::Crystal &crystal,
                         const EnergyModelBase &energy_model) {

  auto asym_mols = crystal.symmetry_unique_molecules();
  m_asym_charges = Vec(crystal.asymmetric_unit().size());
  m_charge_self_energies.resize(asym_mols.size());
  m_electric_field_values.clear();
  m_total_energy = 0.0;

  const auto &partial_charges = energy_model.partial_charges();
  for (size_t i = 0; i < partial_charges.size(); i++) {
    const auto &asymmetric_atom_indices = asym_mols[i].asymmetric_unit_idx();
    const auto &charge_vector = partial_charges[i];

    occ::log::info("Charges used in wolf for molecule {}", i);
    for (int j = 0; j < charge_vector.rows(); j++) {
      occ::log::info("Atom {}: {:12.5f}", j, charge_vector(j));
      m_asym_charges(asymmetric_atom_indices(j)) = charge_vector(j);
    }
  }

  compute_self_energies(crystal);
  compute_wolf_energies(crystal);
}

void WolfSum::compute_self_energies(const crystal::Crystal &crystal) {
  auto asym_mols = crystal.symmetry_unique_molecules();
  for (size_t i = 0; i < asym_mols.size(); i++) {
    const auto &mol = asym_mols[i];
    m_electric_field_values.push_back(Mat3N::Zero(3, mol.size()));
    m_charge_self_energies[i] =
        coulomb_self_energy_asym_charges(mol, m_asym_charges);
  }
}

void WolfSum::compute_wolf_energies(const crystal::Crystal &crystal) {
  auto surrounds = crystal.asymmetric_unit_atom_surroundings(m_params.cutoff);
  Mat3N asym_cart = crystal.to_cartesian(crystal.asymmetric_unit().positions);
  Vec asym_wolf(surrounds.size());

  for (size_t asym_idx = 0; asym_idx < surrounds.size(); asym_idx++) {
    const auto &s = surrounds[asym_idx];
    double qi = m_asym_charges(asym_idx);
    Vec3 pi = asym_cart.col(asym_idx);

    Vec qj(s.size());
    for (int j = 0; j < qj.rows(); j++) {
      qj(j) = m_asym_charges(s.asym_idx(j));
    }

    asym_wolf(asym_idx) =
        wolf_coulomb_energy(qi, pi, qj, s.cart_pos, m_params) *
        units::AU_TO_KJ_PER_MOL;
  }

  auto asym_mols = crystal.symmetry_unique_molecules();
  for (const auto &mol : asym_mols) {
    for (int j = 0; j < mol.size(); j++) {
      m_total_energy += asym_wolf(mol.asymmetric_unit_idx()(j));
    }
  }

  occ::log::debug("Wolf energy ({} asymmetric molecules): {}\n",
                  asym_mols.size(), m_total_energy);
}

double WolfSum::compute_correction(
    const std::vector<double> &charge_energies,
    const std::vector<CEEnergyComponents> &model_energies) const {

  double ecoul_real = 0.0;
  double ecoul_exact_real = 0.0;
  double ecoul_self = 0.0;

  for (size_t i = 0; i < charge_energies.size(); i++) {
    if (model_energies[i].is_computed) {
      ecoul_exact_real += 0.5 * model_energies[i].coulomb_kjmol();
      ecoul_real += 0.5 * charge_energies[i] * units::AU_TO_KJ_PER_MOL;
    }
  }

  for (double self_energy : m_charge_self_energies) {
    ecoul_self += self_energy * units::AU_TO_KJ_PER_MOL;
  }

  return (m_total_energy - ecoul_self - ecoul_real + ecoul_exact_real);
}

} // namespace occ::interaction
