#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/unitcell.h>
#include <occ/xtb/periodic.h>

namespace occ::xtb {

namespace {

// Perpendicular distance from origin to the face of the parallelepiped
// spanned by the other two lattice vectors. For a lattice vector a_i, this
// is V / |a_j × a_k|.
double perpendicular_distance(const Vec3 &ai, const Vec3 &aj, const Vec3 &ak) {
  const double V = std::abs(ai.dot(aj.cross(ak)));
  const double face = aj.cross(ak).norm();
  return V / face;
}

} // namespace

std::vector<LatticeImage> build_lattice_images(const Mat3 &lattice_bohr,
                                               double cutoff_bohr) {
  const Vec3 a = lattice_bohr.col(0);
  const Vec3 b = lattice_bohr.col(1);
  const Vec3 c = lattice_bohr.col(2);
  const double da = perpendicular_distance(a, b, c);
  const double db = perpendicular_distance(b, a, c);
  const double dc = perpendicular_distance(c, a, b);
  // Number of cells we need to extend in each direction so that the
  // farthest face of cell n is still within `cutoff` of the central cell.
  const int na = static_cast<int>(std::ceil(cutoff_bohr / da));
  const int nb = static_cast<int>(std::ceil(cutoff_bohr / db));
  const int nc = static_cast<int>(std::ceil(cutoff_bohr / dc));

  std::vector<LatticeImage> out;
  out.reserve(static_cast<size_t>((2 * na + 1) * (2 * nb + 1) * (2 * nc + 1)));
  for (int i = -na; i <= na; ++i) {
    for (int j = -nb; j <= nb; ++j) {
      for (int k = -nc; k <= nc; ++k) {
        Vec3 t = i * a + j * b + k * c;
        const double n = t.norm();
        // Keep every translation whose face is potentially within cutoff;
        // downstream pair-distance checks prune further.
        if (n - 1e-12 > cutoff_bohr + std::max({da, db, dc})) continue;
        out.push_back({IVec3(i, j, k), t, n});
      }
    }
  }
  std::sort(out.begin(), out.end(),
            [](const LatticeImage &lhs, const LatticeImage &rhs) {
              return lhs.norm < rhs.norm;
            });
  return out;
}

PeriodicSystem PeriodicSystem::from_crystal(const crystal::Crystal &c) {
  PeriodicSystem s;
  // UnitCell::direct() is in Angstroms; convert to Bohr and store as a 3×3
  // with columns = a, b, c.
  const auto &uc = c.unit_cell();
  s.lattice_bohr = uc.direct() * occ::units::ANGSTROM_TO_BOHR;
  // Unit cell atoms (Cartesian, Angstrom).
  const auto &region = c.unit_cell_atoms();
  s.atoms.reserve(region.size());
  const double a2b = occ::units::ANGSTROM_TO_BOHR;
  for (Eigen::Index i = 0; i < region.cart_pos.cols(); ++i) {
    s.atoms.push_back({region.atomic_numbers(i),
                       region.cart_pos(0, i) * a2b,
                       region.cart_pos(1, i) * a2b,
                       region.cart_pos(2, i) * a2b});
  }
  return s;
}

Mat3 PeriodicSystem::reciprocal_bohr() const {
  // 2π × (A^-1)^T, with A having columns a, b, c.
  Mat3 b = lattice_bohr.inverse().transpose();
  return 2.0 * M_PI * b;
}

} // namespace occ::xtb
