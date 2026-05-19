#include <occ/xtb/kpoint_grid.h>
#include <stdexcept>

namespace occ::xtb {

std::vector<KPoint> monkhorst_pack_grid(const Mat3 &reciprocal_bohr, int n1,
                                         int n2, int n3) {
  if (n1 <= 0 || n2 <= 0 || n3 <= 0) {
    throw std::runtime_error("monkhorst_pack_grid: n_i must be positive");
  }
  const Vec3 b1 = reciprocal_bohr.col(0);
  const Vec3 b2 = reciprocal_bohr.col(1);
  const Vec3 b3 = reciprocal_bohr.col(2);
  const double w = 1.0 / static_cast<double>(n1 * n2 * n3);
  std::vector<KPoint> kpts;
  kpts.reserve(static_cast<size_t>(n1 * n2 * n3));
  for (int j1 = 0; j1 < n1; ++j1) {
    for (int j2 = 0; j2 < n2; ++j2) {
      for (int j3 = 0; j3 < n3; ++j3) {
        const double f1 = static_cast<double>(j1) / n1;
        const double f2 = static_cast<double>(j2) / n2;
        const double f3 = static_cast<double>(j3) / n3;
        kpts.push_back({f1 * b1 + f2 * b2 + f3 * b3, w});
      }
    }
  }
  return kpts;
}

} // namespace occ::xtb
