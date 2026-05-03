#include <cmath>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/repulsion.h>
#include <stdexcept>

namespace occ::xtb {

double repulsion_energy(const std::vector<core::Atom> &atoms,
                        const Gfn2Parameters &params) {
  const auto &g = params.globals();
  const double r_exp = 1.0; // rExp; GFN2 uses 1.0
  // Cutoff matches xtb's PBC implementation (40 Bohr) — for molecules this
  // is much larger than the tail.
  constexpr double cutoff2 = 40.0 * 40.0;

  double E = 0.0;
  for (size_t i = 0; i < atoms.size(); ++i) {
    const auto *ei = params.element(atoms[i].atomic_number);
    if (!ei)
      throw std::runtime_error("repulsion_energy: missing parameters for Z=" +
                               std::to_string(atoms[i].atomic_number));
    for (size_t j = 0; j < i; ++j) {
      const auto *ej = params.element(atoms[j].atomic_number);
      if (!ej)
        throw std::runtime_error("repulsion_energy: missing parameters for Z=" +
                                 std::to_string(atoms[j].atomic_number));
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-8)
        continue;
      const double r = std::sqrt(r2);
      const double alpha = std::sqrt(ei->rep_alpha * ej->rep_alpha);
      const double zeff = ei->rep_zeff * ej->rep_zeff;
      const bool light_pair = (atoms[i].atomic_number <= 2) &&
                              (atoms[j].atomic_number <= 2);
      const double k_exp = light_pair ? g.kexplight : g.kexp;
      const double t16 = std::pow(r, k_exp);
      const double t26 = std::exp(-alpha * t16);
      const double t27 = std::pow(r, r_exp);
      E += zeff * t26 / t27;
    }
  }
  return E;
}

} // namespace occ::xtb
