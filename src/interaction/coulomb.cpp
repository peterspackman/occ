#include <occ/core/units.h>
#include <occ/interaction/coulomb.h>

namespace occ::interaction {

double coulomb_energy(Eigen::Ref<const Vec> charges,
                      Eigen::Ref<const Mat3N> positions) {
  double energy = 0.0;
  const int N = charges.rows();
  for (int i = 0; i < N; i++) {
    const Vec3 pi = positions.col(i);
    const double qi = charges(i);
    for (int j = 0; j < i; j++) {
      const double qj = charges(j);
      double r = (positions.col(j) - pi).norm() * occ::units::ANGSTROM_TO_BOHR;
      energy += qi * qj / r;
    }
  }
  return energy;
}

double coulomb_pair_energy(Eigen::Ref<const Vec> charges_a,
                           Eigen::Ref<const Mat3N> positions_a,
                           Eigen::Ref<const Vec> charges_b,
                           Eigen::Ref<const Mat3N> positions_b) {
  double energy = 0.0;
  const int Na = charges_a.rows();
  const int Nb = charges_b.rows();
  for (int i = 0; i < Na; i++) {
    const double qi = charges_a(i);
    for (int j = 0; j < Nb; j++) {
      const double qj = charges_b(j);
      double r = (positions_b.col(j) - positions_a.col(i)).norm() *
                 occ::units::ANGSTROM_TO_BOHR;
      energy += qi * qj / r;
    }
  }
  return energy;
}

Vec3 coulomb_efield(Eigen::Ref<const Vec> charges,
                    Eigen::Ref<const Mat3N> positions,
                    Eigen::Ref<const Vec3> point) {

  Vec3 field = Vec3::Zero();
  for (int i = 0; i < charges.rows(); i++) {
    Vec3 r = (point - positions.col(i)) * occ::units::ANGSTROM_TO_BOHR;
    double rmag = r.norm();
    field += charges(i) * r / (rmag * rmag * rmag);
  }
  return field;
}

std::pair<Mat3N, Mat3N> coulomb_pair_efield(
    Eigen::Ref<const Vec> charges_a, Eigen::Ref<const Mat3N> positions_a,
    Eigen::Ref<const Vec> charges_b, Eigen::Ref<const Mat3N> positions_b) {
  const int Na = charges_a.rows();
  const int Nb = charges_b.rows();
  Mat3N efield_a = Mat3N::Zero(3, Na);
  Mat3N efield_b = Mat3N::Zero(3, Nb);
  for (int i = 0; i < Na; i++) {
    efield_a.col(i) =
        coulomb_efield(charges_b, positions_b, positions_a.col(i));
  }

  for (int i = 0; i < Nb; i++) {
    efield_b.col(i) =
        coulomb_efield(charges_a, positions_a, positions_b.col(i));
  }

  return {efield_a, efield_b};
}

double
coulomb_interaction_energy_asym_charges(const occ::core::Dimer &dimer,
                                        Eigen::Ref<const Vec> charges_asym) {
  const auto &pos_a = dimer.a().positions();
  const auto &pos_b = dimer.b().positions();
  const auto &asym_atom_a = dimer.a().asymmetric_unit_idx();
  const auto &asym_atom_b = dimer.b().asymmetric_unit_idx();

  auto idx_to_charge = [&charges_asym](int idx) { return charges_asym(idx); };
  Vec charges_a = asym_atom_a.unaryExpr(idx_to_charge);
  Vec charges_b = asym_atom_b.unaryExpr(idx_to_charge);

  return coulomb_pair_energy(charges_a, pos_a, charges_b, pos_b);
}

double coulomb_self_energy_asym_charges(const occ::core::Molecule &mol,
                                        Eigen::Ref<const Vec> charges_asym) {
  const auto &pos = mol.positions();
  const auto &asym_atom = mol.asymmetric_unit_idx();
  auto idx_to_charge = [&charges_asym](int idx) { return charges_asym(idx); };
  Vec charges = asym_atom.unaryExpr(idx_to_charge);
  return coulomb_energy(charges, pos);
}

std::pair<Mat3N, Mat3N>
coulomb_efield_asym_charges(const occ::core::Dimer &dimer,
                            Eigen::Ref<const Vec> charges_asym) {
  const auto &pos_a = dimer.a().positions();
  const auto &pos_b = dimer.b().positions();
  const auto &asym_atom_a = dimer.a().asymmetric_unit_idx();
  const auto &asym_atom_b = dimer.b().asymmetric_unit_idx();

  auto idx_to_charge = [&charges_asym](int idx) { return charges_asym(idx); };
  Vec charges_a = asym_atom_a.unaryExpr(idx_to_charge);
  Vec charges_b = asym_atom_b.unaryExpr(idx_to_charge);

  return coulomb_pair_efield(charges_a, pos_a, charges_b, pos_b);
}

} // namespace occ::interaction
