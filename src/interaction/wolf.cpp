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
