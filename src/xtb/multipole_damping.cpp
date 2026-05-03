#include <array>
#include <cmath>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/multipole_damping.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// xtb's multiRad table (Bohr); from src/xtb/data.f90. NOTE: stored directly
// in atomic units in xtb (no aatoau scaling).
constexpr std::array<double, 87> multi_rad_bohr = {
    0.0,
    1.4, 3.0,                                              // H, He
    5.0, 5.0, 5.0, 3.0, 1.9, 1.8, 2.4, 5.0,                // Li-Ne
    5.0, 5.0, 5.0, 3.9, 2.1, 3.1, 2.5, 5.0,                // Na-Ar
    5.0, 5.0,                                              // K, Ca
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,      // Sc-Zn
    5.0, 5.0, 5.0, 3.9, 4.0, 5.0,                          // Ga-Kr
    5.0, 5.0,                                              // Rb, Sr
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,      // Y-Cd
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,                          // In-Xe
    5.0, 5.0,                                              // Cs, Ba
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,                     // La-Eu
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,                     // Gd-Yb
    5.0, 5.0, 5.0, 5.0, 5.0,                               // Lu-Re
    5.0, 5.0, 5.0, 5.0, 5.0,                               // Os-Hg
    5.0, 5.0, 5.0, 5.0, 5.0, 5.0,                          // Tl-Rn
};

// Per-element valenceCN, from xtb's data.f90.
constexpr std::array<double, 87> valence_cn = {
    0.0,
    1.0, 1.0,                                                  // H, He
    1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0,                    // Li-Ne
    1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,                    // Na-Ar
    1.0, 2.0,                                                  // K, Ca
    4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 2.0,          // Sc-Zn
    3.0, 3.0, 3.0, 3.0, 1.0, 1.0,                              // Ga-Kr
    1.0, 2.0,                                                  // Rb, Sr
    4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 2.0,          // Y-Cd
    3.0, 3.0, 3.0, 3.0, 1.0, 1.0,                              // In-Xe
    1.0, 2.0,                                                  // Cs, Ba
    4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,                         // La-Eu
    6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,                         // Gd-Yb
    6.0, 4.0, 6.0, 6.0, 6.0,                                   // Lu-Re
    6.0, 6.0, 4.0, 4.0, 2.0,                                   // Os-Hg
    3.0, 3.0, 3.0, 3.0, 1.0, 1.0,                              // Tl-Rn
};

double multi_rad(int z) {
  if (z < 1 || z > 86) {
    throw std::runtime_error("multipole_damping: unsupported Z=" +
                             std::to_string(z));
  }
  return multi_rad_bohr[z];
}

double val_cn(int z) {
  if (z < 1 || z > 86) {
    throw std::runtime_error("multipole_damping: unsupported Z=" +
                             std::to_string(z));
  }
  return valence_cn[z];
}

} // namespace

Vec multipole_radii(const std::vector<core::Atom> &atoms, const Vec &cn,
                    const Gfn2Parameters &params) {
  const auto &g = params.globals();
  const double rmax = g.aesrmax;
  const double expo = g.aesexp;
  const double shift = g.aesshift;
  const int n = static_cast<int>(atoms.size());
  Vec r(n);
  for (int i = 0; i < n; ++i) {
    const int z = atoms[i].atomic_number;
    const double rco = multi_rad(z);
    const double t = cn(i) - val_cn(z) - shift;
    r(i) = rco + (rmax - rco) / (1.0 + std::exp(-expo * t));
  }
  return r;
}

DampedCoulomb damped_multipole_coulomb(const std::vector<core::Atom> &atoms,
                                       const Vec &mp_radii,
                                       const Gfn2Parameters &params) {
  const auto &g = params.globals();
  const double k3 = g.aesdmp3;
  const double k5 = g.aesdmp5;
  const int n = static_cast<int>(atoms.size());
  DampedCoulomb out;
  out.gab3 = Mat::Zero(n, n);
  out.gab5 = Mat::Zero(n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r = std::sqrt(r2);
      const double rinv = 1.0 / r;
      const double rco = 0.5 * (mp_radii(i) + mp_radii(j));
      const double rcoinvr = rco * rinv;
      const double damp3 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k3));
      const double damp5 = 1.0 / (1.0 + 6.0 * std::pow(rcoinvr, k5));
      const double g3 = damp3 * std::pow(rinv, 3.0);
      const double g5 = damp5 * std::pow(rinv, 5.0);
      out.gab3(i, j) = g3;
      out.gab3(j, i) = g3;
      out.gab5(i, j) = g5;
      out.gab5(j, i) = g5;
    }
  }
  return out;
}

} // namespace occ::xtb
