#include <cmath>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/periodic.h>
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

RepulsionEnergyGradient
repulsion_energy_and_gradient(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params) {
  // dE_pair/dr = -(α · k_exp · r^k_exp + r_exp) · E_pair · (r_i − r_j)/r²
  // (matches xtb's `dG = -(alpha*t16*kExp + repData%rExp) * dE * rij/r2`).
  const auto &g = params.globals();
  const double r_exp = 1.0;
  constexpr double cutoff2 = 40.0 * 40.0;

  RepulsionEnergyGradient out;
  out.energy = 0.0;
  out.gradient = Mat3N::Zero(3, atoms.size());

  for (size_t i = 0; i < atoms.size(); ++i) {
    const auto *ei = params.element(atoms[i].atomic_number);
    if (!ei)
      throw std::runtime_error("repulsion_energy_and_gradient: missing Z=" +
                               std::to_string(atoms[i].atomic_number));
    for (size_t j = 0; j < i; ++j) {
      const auto *ej = params.element(atoms[j].atomic_number);
      if (!ej)
        throw std::runtime_error("repulsion_energy_and_gradient: missing Z=" +
                                 std::to_string(atoms[j].atomic_number));
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-8) continue;
      const double r = std::sqrt(r2);
      const double alpha = std::sqrt(ei->rep_alpha * ej->rep_alpha);
      const double zeff = ei->rep_zeff * ej->rep_zeff;
      const bool light_pair = (atoms[i].atomic_number <= 2) &&
                              (atoms[j].atomic_number <= 2);
      const double k_exp = light_pair ? g.kexplight : g.kexp;
      const double t16 = std::pow(r, k_exp);
      const double t26 = std::exp(-alpha * t16);
      const double t27 = std::pow(r, r_exp);
      const double e_pair = zeff * t26 / t27;
      out.energy += e_pair;
      // dE/dr scalar: factor multiplying (r_i - r_j)/r².
      const double dscal = -(alpha * t16 * k_exp + r_exp) * e_pair / r2;
      // Atom i: +dscal · (r_i − r_j); Atom j: −dscal · (r_i − r_j).
      out.gradient(0, i) += dscal * dx;
      out.gradient(1, i) += dscal * dy;
      out.gradient(2, i) += dscal * dz;
      out.gradient(0, j) -= dscal * dx;
      out.gradient(1, j) -= dscal * dy;
      out.gradient(2, j) -= dscal * dz;
    }
  }
  return out;
}

double repulsion_energy_periodic(
    const std::vector<core::Atom> &atoms, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations) {
  const auto &g = params.globals();
  const double r_exp = 1.0;
  constexpr double cutoff2 = 40.0 * 40.0;

  // E_per_cell = ½ Σ_T Σ_{i,j} V(r_i - r_j - T), excluding (T=0, i=j).
  double E = 0.0;
  for (size_t i = 0; i < atoms.size(); ++i) {
    const auto *ei = params.element(atoms[i].atomic_number);
    if (!ei)
      throw std::runtime_error("repulsion_energy_periodic: missing Z=" +
                               std::to_string(atoms[i].atomic_number));
    const bool light_i = atoms[i].atomic_number <= 2;
    for (size_t j = 0; j < atoms.size(); ++j) {
      const auto *ej = params.element(atoms[j].atomic_number);
      if (!ej)
        throw std::runtime_error("repulsion_energy_periodic: missing Z=" +
                                 std::to_string(atoms[j].atomic_number));
      const double alpha = std::sqrt(ei->rep_alpha * ej->rep_alpha);
      const double zeff = ei->rep_zeff * ej->rep_zeff;
      const bool light_j = atoms[j].atomic_number <= 2;
      const double k_exp = (light_i && light_j) ? g.kexplight : g.kexp;
      for (const auto &im : translations) {
        const bool central =
            im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0;
        if (central && i == j) continue;
        const double dx = atoms[i].x - (atoms[j].x + im.t_bohr.x());
        const double dy = atoms[i].y - (atoms[j].y + im.t_bohr.y());
        const double dz = atoms[i].z - (atoms[j].z + im.t_bohr.z());
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2 || r2 < 1e-8) continue;
        const double r = std::sqrt(r2);
        const double t16 = std::pow(r, k_exp);
        const double t26 = std::exp(-alpha * t16);
        const double t27 = std::pow(r, r_exp);
        E += zeff * t26 / t27;
      }
    }
  }
  return 0.5 * E;
}

} // namespace occ::xtb
