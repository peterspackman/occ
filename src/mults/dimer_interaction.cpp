#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <occ/core/units.h>
#include <occ/mults/dimer_interaction.h>
#include <set>
#include <stdexcept>

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

const std::vector<ShortRangeModel> &short_range_model_registry() {
  static const std::vector<ShortRangeModel> registry = {
      {"w99", "Williams-1999 exp-6, NEIGHCRYS-typed (H_W1, C_W3, ...)",
       "neighcrys", {"williams"}},
      {"fit",
       "FIT (Williams/Cox) exp-6, typed with the H_F1/H_F2 polar-hydrogen split",
       "neighcrys-fit", {}},
      {"williams-de", "Williams DE exp-6, element-based (no atom typing)",
       "none", {"de", "williams_de"}},
      {"none", "no short-range parameters", "none", {}},
  };
  return registry;
}

std::vector<std::string> short_range_model_names() {
  std::vector<std::string> names;
  for (const auto &m : short_range_model_registry()) {
    names.push_back(m.name);
    names.insert(names.end(), m.aliases.begin(), m.aliases.end());
  }
  return names;
}

namespace {
std::string to_lower(const std::string &s) {
  std::string out(s.size(), '\0');
  std::transform(s.begin(), s.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}
} // namespace

const ShortRangeModel &short_range_model_from_string(const std::string &name) {
  const auto lower = to_lower(name);
  for (const auto &m : short_range_model_registry()) {
    if (m.name == lower)
      return m;
    if (std::find(m.aliases.begin(), m.aliases.end(), lower) != m.aliases.end())
      return m;
  }
  throw std::runtime_error(
      fmt::format("Unknown short-range model '{}' (available: {})", name,
                  fmt::join(short_range_model_names(), ", ")));
}

const ShortRangeModel &
short_range_model_for_model_name(const std::string &model_name) {
  const auto lower = to_lower(model_name);
  // Only the typed sets are named explicitly inside a compound model name;
  // everything else ("dma", "williams", "dma-b3lyp", ...) means the
  // element-based table. Precedence is fixed here rather than taken from
  // registry order, which is presentation order.
  for (const char *name : {"fit", "w99"}) {
    if (lower.find(name) != std::string::npos)
      return short_range_model_from_string(name);
  }
  return short_range_model_from_string("williams-de");
}

ForceFieldParams make_force_field(const ShortRangeModel &model) {
  if (model.name == "fit")
    return fit_force_field();
  if (model.name == "w99")
    return williams_typed_force_field();
  return williams_de_force_field();
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
