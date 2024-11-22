#include <occ/core/eem.h>
#include <occ/core/units.h>

namespace occ::core::charges {
namespace impl {
std::pair<double, double> params(int atomic_number) {
  switch (atomic_number) {
  case 1:
    return {0.20606, 1.31942};
  case 3:
    return {0.36237, 0.65932};
  case 5:
    return {0.36237, 0.65932};
  case 6:
    return {0.36237, 0.65932};
  case 7:
    return {0.49279, 0.69038};
  case 8:
    return {0.73013, 1.08856};
  case 9:
    return {0.72052, 1.45328};
  case 11:
    return {0.36237, 0.65932};
  case 12:
    return {0.36237, 0.65932};
  case 14:
    return {0.36237, 0.65932};
  case 15:
    return {0.36237, 0.65932};
  case 16:
    return {0.62020, 0.41280};
  case 17:
    return {0.36237, 0.65932};
  case 19:
    return {0.36237, 0.65932};
  case 20:
    return {0.36237, 0.65932};
  case 26:
    return {0.36237, 0.65932};
  case 29:
    return {0.36237, 0.65932};
  case 30:
    return {0.36237, 0.65932};
  case 35:
    return {0.70052, 1.09108};
  default:
    return {0.20606, 1.31942};
  }
};
} // namespace impl

occ::Vec eem_partial_charges(const occ::IVec &atomic_numbers,
                             const occ::Mat3N &positions, double charge) {
  size_t N = atomic_numbers.rows();
  occ::Vec A(N), B(N);

  double a, b;
  for (size_t i = 0; i < N; i++) {
    std::tie(a, b) = impl::params(atomic_numbers(i));
    A(i) = a;
    B(i) = b;
  }

  occ::Mat M(N + 1, N + 1);
  M.col(N).setConstant(-1.0);
  M.row(N).setConstant(1.0);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = i + 1; j < N; j++) {
      double norm = occ::units::ANGSTROM_TO_BOHR *
                    (positions.col(i) - positions.col(j)).norm();
      if (norm != 0.0)
        M(i, j) = 1.0 / norm;
      else
        M(i, j) = 0.0;
      M(j, i) = M(i, j);
    }
    M(i, i) = B(i);
  }
  M(N, N) = 0.0;

  occ::Vec y = occ::Vec(N + 1);
  y.topRows(N).array() = -A.array();
  y(N) = charge;
  return M.householderQr().solve(y).topRows(N);
}

} // namespace occ::core::charges
