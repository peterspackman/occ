#include <cmath>
#include <occ/core/quasirandom.h>

namespace occ::core {

double phi(size_t d) {
  int iterations = 30;
  double x = 2.0;
  for (int i = 0; i < iterations; i++) {
    x = std::pow(1 + x, 1.0 / (d + 1.0));
  }
  return x;
}

Vec alpha(size_t ndims) {
  double g = phi(ndims);
  Vec a(ndims);
  for (int i = 0; i < ndims; i++) {
    a(i) = std::fmod(std::pow(1 / g, i + 1.0), 1.0);
  }
  return a;
}

Mat quasirandom_kgf(size_t ndims, size_t count, size_t seed) {
  constexpr static double offset = 0.5;
  Vec a = alpha(ndims).array();
  Mat result(ndims, count);

  for (int i = 0; i < count; i++) {
    result.col(i) = (a.array() * (seed + i + 1)).unaryExpr([](double x) {
      return std::fmod(x + offset, 1.0);
    });
  }
  return result;
}

} // namespace occ::core
