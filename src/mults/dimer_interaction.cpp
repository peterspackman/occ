#include <algorithm>
#include <cmath>
#include <occ/core/units.h>
#include <occ/mults/dimer_interaction.h>
#include <set>

namespace occ::mults {

DimerInteractionEnergy
dimer_interaction_energy(const MoleculeMultipoles &a,
                         const MoleculeMultipoles &b,
                         const ForceFieldParams &ff,
                         const MultipoleInteractions::Config &elec_config) {
  DimerInteractionEnergy result;
  const Eigen::Index na = a.positions.cols(), nb = b.positions.cols();

  // --- electrostatics: multipole-multipole over all site pairs (Bohr, Hartree)
  MultipoleInteractions elec(elec_config);
  double e_elec_au = 0.0;
  for (Eigen::Index i = 0; i < na; ++i) {
    const Vec3 pi = a.positions.col(i);
    for (Eigen::Index j = 0; j < nb; ++j) {
      e_elec_au += elec.compute_interaction_energy(
          a.multipoles[i], pi, b.multipoles[j], b.positions.col(j));
    }
  }
  result.electrostatic = occ::units::AU_TO_KJ_PER_MOL * e_elec_au;

  // --- exp-6 repulsion/dispersion over atom pairs (Angstrom, kJ/mol); typed
  // NEIGHCRYS lookup when type codes are present, element-based otherwise.
  const bool typed = ff.use_short_range_typing() && !a.type_codes.empty() &&
                     !b.type_codes.empty();
  for (Eigen::Index i = 0; i < na; ++i) {
    const int za = a.atomic_numbers[i];
    const int ta = typed ? a.type_codes[i] : 0;
    const Vec3 pi = a.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
    for (Eigen::Index j = 0; j < nb; ++j) {
      const int zb = b.atomic_numbers[j];
      BuckinghamParams p;
      if (typed && ta > 0 && b.type_codes[j] > 0) {
        // both sites classified -> typed lookup (element fallback inside)
        p = ff.get_buckingham_for_types(ta, b.type_codes[j]);
      } else {
        if (!ff.has_buckingham(za, zb))
          continue; // unparameterised element pair (caller's responsibility)
        p = ff.get_buckingham(za, zb);
      }
      const Vec3 pj = b.positions.col(j) * occ::units::BOHR_TO_ANGSTROM;
      const double r = (pj - pi).norm();
      const double r6 = std::pow(r, 6);
      result.repulsion += p.A * std::exp(-p.B * r);
      result.dispersion += -p.C / r6;
    }
  }
  return result;
}

ForceFieldParams williams_de_force_field() {
  ForceFieldParams ff;
  for (const auto &[zz, params] : ForceFieldParams::williams_de_params())
    ff.set_buckingham(zz.first, zz.second, params);
  return ff;
}

ForceFieldParams fit_force_field() {
  ForceFieldParams ff = williams_de_force_field(); // element-based fallback
  ff.set_typed_buckingham(ForceFieldParams::fit_typed_params());
  ff.set_use_williams_atom_typing(true);
  return ff;
}

ForceFieldParams williams_typed_force_field() {
  ForceFieldParams ff = williams_de_force_field(); // element-based fallback
  ff.set_typed_buckingham(ForceFieldParams::williams_typed_params());
  ff.set_use_williams_atom_typing(true);
  return ff;
}

std::vector<std::pair<int, int>>
missing_exp6_pairs(const std::vector<int> &elements,
                   const ForceFieldParams &ff) {
  const std::set<int> uniq(elements.begin(), elements.end());
  std::set<std::pair<int, int>> missing;
  for (int za : uniq)
    for (int zb : uniq)
      if (!ff.has_buckingham(za, zb))
        missing.insert({std::min(za, zb), std::max(za, zb)});
  return {missing.begin(), missing.end()};
}

} // namespace occ::mults
