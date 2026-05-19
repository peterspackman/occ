#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/units.h>
#include <occ/scrf/cosmo_kernel.h>
#include <occ/solvent/cosmo.h>

namespace occ::solvent {

namespace cosmo {

Vec solvation_radii(const IVec &nums) {
  // angstroms
  occ::Vec result(nums.rows());
  static const double radii[17] = {1.300, 1.638, 1.404, 1.053,  2.0475, 2.00,
                                   1.830, 1.720, 1.720, 1.8018, 1.755,  1.638,
                                   1.404, 2.457, 2.106, 2.160,  2.05};

  for (size_t i = 0; i < nums.rows(); i++) {
    int n = nums(i);
    double r = 2.223;
    if (n <= 17 && n > 0)
      r = radii[n - 1];
    result(i) = r;
  }
  return result * occ::units::ANGSTROM_TO_BOHR;
}

} // namespace cosmo

COSMO::Result COSMO::operator()(const Mat3N &positions, const Vec &areas,
                                const Vec &charges) const {
  // Shared kernel: same A matrix construction (1.07·√(4π/S_i) diagonal,
  // 1/|r_i - r_j| off-diagonal) as the xTB-side CPCM-X / SMD ES path.
  COSMO::Result res;
  Mat A = occ::scrf::detail::build_cosmo_A(positions, areas);
  res.initial = -surface_charge(charges);
  res.converged = A.lu().solve(res.initial);
  res.energy = -0.5 * res.initial.dot(res.converged);
  return res;
}

} // namespace occ::solvent
