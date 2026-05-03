#include <array>
#include <cmath>
#include <occ/core/units.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/periodic.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Pyykko & Atsumi covalent radii (Chem. Eur. J. 15, 188-197, 2009). Values
// for metals reduced by 10%; same table xtb uses in src/param/covalentradd3.f90.
// Stored in Bohr × 4/3 — i.e. directly the (R_cov_A + R_cov_B) factor that
// goes into the GFN counting function.
//
// Index by atomic number Z (0 entry is a placeholder).
constexpr double angstrom_to_bohr = occ::units::ANGSTROM_TO_BOHR;
constexpr double scale = 4.0 / 3.0 * angstrom_to_bohr;

constexpr std::array<double, 87> covalent_rad_d3_bohr = {
    0.0,
    0.32 * scale, 0.46 * scale,                                 // H, He
    1.20 * scale, 0.94 * scale, 0.77 * scale, 0.75 * scale,
    0.71 * scale, 0.63 * scale, 0.64 * scale, 0.67 * scale,     // Li-Ne
    1.40 * scale, 1.25 * scale, 1.13 * scale, 1.04 * scale,
    1.10 * scale, 1.02 * scale, 0.99 * scale, 0.96 * scale,     // Na-Ar
    1.76 * scale, 1.54 * scale,                                 // K, Ca
    1.33 * scale, 1.22 * scale, 1.21 * scale, 1.10 * scale, 1.07 * scale,
    1.04 * scale, 1.00 * scale, 0.99 * scale, 1.01 * scale, 1.09 * scale, // Sc-Zn
    1.12 * scale, 1.09 * scale, 1.15 * scale, 1.10 * scale,
    1.14 * scale, 1.17 * scale,                                 // Ga-Kr
    1.89 * scale, 1.67 * scale,                                 // Rb, Sr
    1.47 * scale, 1.39 * scale, 1.32 * scale, 1.24 * scale, 1.15 * scale,
    1.13 * scale, 1.13 * scale, 1.08 * scale, 1.15 * scale, 1.23 * scale, // Y-Cd
    1.28 * scale, 1.26 * scale, 1.26 * scale, 1.23 * scale,
    1.32 * scale, 1.31 * scale,                                 // In-Xe
    2.09 * scale, 1.76 * scale,                                 // Cs, Ba
    1.62 * scale, 1.47 * scale, 1.58 * scale, 1.57 * scale, 1.56 * scale,
    1.55 * scale, 1.51 * scale,                                 // La-Eu
    1.52 * scale, 1.51 * scale, 1.50 * scale, 1.49 * scale, 1.49 * scale,
    1.48 * scale, 1.53 * scale,                                 // Gd-Yb
    1.46 * scale, 1.37 * scale, 1.31 * scale, 1.23 * scale, 1.18 * scale, // Lu-Re
    1.16 * scale, 1.11 * scale, 1.12 * scale, 1.13 * scale, 1.32 * scale, // Os-Hg
    1.30 * scale, 1.30 * scale, 1.36 * scale, 1.31 * scale,
    1.38 * scale, 1.42 * scale,                                 // Tl-Rn
};

double cov_rad(int z) {
  if (z < 1 || z > 86) {
    throw std::runtime_error(
        "GFN coordination number: unsupported element Z=" + std::to_string(z));
  }
  return covalent_rad_d3_bohr[z];
}

// Counting function: 1 / (1 + exp(-k (r0/r - 1)))
double exp_count(double k, double r, double r0) {
  return 1.0 / (1.0 + std::exp(-k * (r0 / r - 1.0)));
}

// gfn flavor: expCount(k, r, r0) * expCount(2k, r, r0 + 2)
//
// xtb uses k = 10 for GFN1/GFN2. The trailing "+ 2" is in *Bohr*.
double gfn_count(double r, double r0) {
  constexpr double k = 10.0;
  return exp_count(k, r, r0) * exp_count(2.0 * k, r, r0 + 2.0);
}

} // namespace

Vec gfn_coordination_numbers(const std::vector<core::Atom> &atoms) {
  const int n = static_cast<int>(atoms.size());
  Vec cn = Vec::Zero(n);
  // Match xtb: hard cutoff at 40 Bohr.
  constexpr double cutoff = 40.0;
  constexpr double cutoff2 = cutoff * cutoff;

  for (int i = 0; i < n; ++i) {
    const double xi = atoms[i].x, yi = atoms[i].y, zi = atoms[i].z;
    const double rc_i = cov_rad(atoms[i].atomic_number);
    for (int j = 0; j < i; ++j) {
      const double dx = xi - atoms[j].x;
      const double dy = yi - atoms[j].y;
      const double dz = zi - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2) continue;
      const double r = std::sqrt(r2);
      const double r0 = rc_i + cov_rad(atoms[j].atomic_number);
      const double c = gfn_count(r, r0);
      cn(i) += c;
      cn(j) += c;
    }
  }
  return cn;
}

Vec gfn_coordination_numbers_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<LatticeImage> &translations) {
  const int n = static_cast<int>(atoms.size());
  Vec cn = Vec::Zero(n);
  constexpr double cutoff = 40.0;
  constexpr double cutoff2 = cutoff * cutoff;

  for (int i = 0; i < n; ++i) {
    const double xi = atoms[i].x, yi = atoms[i].y, zi = atoms[i].z;
    const double rc_i = cov_rad(atoms[i].atomic_number);
    for (int j = 0; j < n; ++j) {
      const double rxj = atoms[j].x, ryj = atoms[j].y, rzj = atoms[j].z;
      const double rc_j = cov_rad(atoms[j].atomic_number);
      const double r0 = rc_i + rc_j;
      for (const auto &im : translations) {
        // Skip the (T=0, i=j) self-pair.
        const bool central = im.hkl(0) == 0 && im.hkl(1) == 0 && im.hkl(2) == 0;
        if (central && j == i) continue;
        const double dx = xi - (rxj + im.t_bohr.x());
        const double dy = yi - (ryj + im.t_bohr.y());
        const double dz = zi - (rzj + im.t_bohr.z());
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cutoff2) continue;
        const double r = std::sqrt(r2);
        cn(i) += gfn_count(r, r0);
      }
    }
  }
  return cn;
}

} // namespace occ::xtb
