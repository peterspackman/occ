#include <cmath>
#include <occ/xtb/ewald_common.h>

namespace occ::xtb {

namespace {
constexpr double kSqrtPi = 1.7724538509055160273;
}

double auto_ewald_alpha(double volume_bohr3) {
  return kSqrtPi / std::cbrt(volume_bohr3);
}

double ewald_real_cutoff(double alpha, double tol) {
  const double x = std::sqrt(-std::log(tol));
  return x / alpha + 1.0;  // +1 Bohr safety margin
}

double ewald_recip_cutoff(double alpha, double tol) {
  const double x = std::sqrt(-std::log(tol));
  return 2.0 * alpha * x;
}

std::vector<Vec3> enumerate_g_vectors(const Mat3 &reciprocal_bohr,
                                       double recip_cutoff) {
  const Vec3 b1 = reciprocal_bohr.col(0);
  const Vec3 b2 = reciprocal_bohr.col(1);
  const Vec3 b3 = reciprocal_bohr.col(2);
  auto bound = [&](const Vec3 &b) {
    return static_cast<int>(std::ceil(recip_cutoff / b.norm())) + 1;
  };
  const int n1 = bound(b1);
  const int n2 = bound(b2);
  const int n3 = bound(b3);
  std::vector<Vec3> out;
  out.reserve(static_cast<size_t>((2 * n1 + 1) * (2 * n2 + 1) * (2 * n3 + 1)));
  const double cutoff2 = recip_cutoff * recip_cutoff;
  for (int i = -n1; i <= n1; ++i) {
    for (int j = -n2; j <= n2; ++j) {
      for (int k = -n3; k <= n3; ++k) {
        Vec3 G = i * b1 + j * b2 + k * b3;
        const double g2 = G.squaredNorm();
        if (g2 < 1e-20 || g2 > cutoff2) continue;
        out.push_back(G);
      }
    }
  }
  return out;
}

} // namespace occ::xtb
