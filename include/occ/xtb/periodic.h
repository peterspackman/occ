#pragma once
#include <Eigen/Geometry>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::crystal {
class Crystal;
}

namespace occ::xtb {

// A single lattice translation T = n1·a + n2·b + n3·c.
struct LatticeImage {
  IVec3 hkl;       // (n1, n2, n3) — integer lattice indices
  Vec3 t_bohr;     // T in Bohr
  double norm;     // |T| in Bohr (cached for sorting / pruning)
};

// All translation vectors needed for a real-space sum with cutoff `R`,
// given an atomic span (max distance between two atoms in the central cell)
// `Rmax_pair`. Includes T = (0,0,0). Sorted by |T| ascending.
//
// `lattice_bohr.cols()` are the lattice vectors a, b, c in Bohr.
std::vector<LatticeImage> build_lattice_images(const Mat3 &lattice_bohr,
                                               double cutoff_bohr);

// A periodic system represented by its central-cell atoms (in Bohr) plus
// the 3×3 lattice in Bohr. Convenience wrapper around occ::crystal::Crystal.
struct PeriodicSystem {
  std::vector<core::Atom> atoms; // unit-cell atoms (Bohr)
  Mat3 lattice_bohr;             // columns are a, b, c

  // Build a PeriodicSystem from an occ::crystal::Crystal.
  static PeriodicSystem from_crystal(const occ::crystal::Crystal &c);

  inline int num_atoms() const { return static_cast<int>(atoms.size()); }
  inline double volume() const {
    return std::abs(lattice_bohr.col(0).dot(
        lattice_bohr.col(1).cross(lattice_bohr.col(2))));
  }
  // Reciprocal lattice 2π · (A^-1)^T (columns are b1, b2, b3 in 1/Bohr).
  Mat3 reciprocal_bohr() const;
};

} // namespace occ::xtb
