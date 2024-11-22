#include <occ/core/log.h>
#include <occ/core/macros.h>
#include <occ/sht/wigner3j.h>

namespace occ::sht {

namespace impl {

double w3a(double l1, double l2, double l3, double m1, double /*m2*/,
           double /*m3*/) {
  double T1 = l1 * l1 - std::pow(l2 - l3, 2.0);
  double T2 = std::pow(l2 + l3 + 1.0, 2.0) - l1 * l1;
  double T3 = l1 * l1 - m1 * m1;
  return std::sqrt(T1 * T2 * T3);
}

double w3b(double l1, double l2, double l3, double m1, double m2, double m3) {
  double T1 = -(2.0 * l1 + 1.0);
  double T2 = l2 * (l2 + 1.0) * m1;
  double T3 = l3 * (l3 + 1.0) * m1;
  double T4 = l1 * (l1 + 1.0) * (m3 - m2);
  return T1 * (T2 - T3 - T4);
}

OCC_ALWAYS_INLINE double alpha_term_l1_zero(double l1, double l2, double l3,
                                            double m1, double m2, double m3) {
  return -(m3 - m2 + 2.0 * w3b(l1, l2, l3, m1, m2, m3)) /
         w3a(1.0, l2, l3, m1, m2, m3);
}

OCC_ALWAYS_INLINE double alpha_term(double l1, double l2, double l3, double m1,
                                    double m2, double m3) {
  return -w3b(l1, l2, l3, m1, m2, m3) /
         (l1 * w3a(l1 + 1.0, l2, l3, m1, m2, m3));
}

OCC_ALWAYS_INLINE double alpha_term_backward(double l1, double l2, double l3,
                                             double m1, double m2, double m3) {
  return -w3b(l1, l2, l3, m1, m2, m3) /
         ((l1 + 1.0) * w3a(l1, l2, l3, m1, m2, m3));
}

OCC_ALWAYS_INLINE double beta_term(double l1, double l2, double l3, double m1,
                                   double m2, double m3) {
  return -(l1 + 1.0) * w3a(l1, l2, l3, m1, m2, m3) /
         (l1 * w3a(l1 + 1.0, l2, l3, m1, m2, m3));
}

OCC_ALWAYS_INLINE double beta_term_backward(double l1, double l2, double l3,
                                            double m1, double m2, double m3) {
  return -l1 * w3a(l1 + 1.0, l2, l3, m1, m2, m3) /
         ((l1 + 1.0) * w3a(l1, l2, l3, m1, m2, m3));
}

} // namespace impl

Vec wigner3j(double l2, double l3, double m1, double m2, double m3) {
  // compute the numeric limits of double precision.
  const double huge = std::sqrt(std::numeric_limits<double>::max() / 20.0);
  const double huge_sentinel = std::sqrt(huge);
  constexpr double tiny = std::numeric_limits<double>::min();
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const double tiny_sentinel = std::sqrt(tiny + eps);

  if (!(std::fabs(m1 + m2 + m3) < eps && std::fabs(m2) <= l2 + eps &&
        std::fabs(m3) <= l3 + eps))
    return Vec::Zero(1);

  double l1min = std::max(std::fabs(l2 - l3), std::fabs(m1));
  double l1max = l2 + l3;

  int size = static_cast<int>(std::floor(l1max - l1min + 1.0 + eps));
  Vec result = Vec::Zero(size);
  const bool l1min_zero = l1min == 0.0;

  if (size == 1) {
    double exponent = std::floor(std::fabs(l2 + m2 - l3 + m3));
    result(0) = std::pow(-1.0, exponent) / std::sqrt(l1min + l2 + l3 + 1.0);

  } else {
    result(0) = tiny_sentinel;
    double an, l1 = l1min;

    if (l1min_zero)
      an = impl::alpha_term_l1_zero(l1, l2, l3, m1, m2, m3);
    else
      an = impl::alpha_term(l1min, l2, l3, m1, m2, m3);

    result(1) = an * result(0);

    if (size > 2) {
      result(0) = tiny_sentinel;

      double prev_an, beta;
      if (l1min_zero)
        an = impl::alpha_term_l1_zero(l1, l2, l3, m1, m2, m3);
      else
        an = impl::alpha_term(l1min, l2, l3, m1, m2, m3);

      result(1) = an * result(0);

      int i = 1;
      bool av = false;
      do {
        i++;
        prev_an = an;
        l1 += 1.0;

        an = impl::alpha_term(l1, l2, l3, m1, m2, m3);
        beta = impl::beta_term(l1, l2, l3, m1, m2, m3);

        result(i) = an * result(i - 1) + beta * result(i - 2);

        if (std::fabs(result(i)) > huge_sentinel) {
          occ::log::debug("renormalized forward recursion in wigner3j");
          result.head(i) /= huge_sentinel;
        }

        if (av)
          break;
        if (std::fabs(an) - std::fabs(prev_an) > 0.0)
          av = true;
      } while (i < (size - 1));

      if (i != size - 1) {
        double l1_mid_minus_one = result(i - 2);
        double l1_mid = result(i - 1);
        double l1_mid_plus_one = result(i);

        result(size - 1) = tiny_sentinel;

        l1 = l1max;
        an = impl::alpha_term_backward(l1, l2, l3, m1, m2, m3);
        result(size - 2) = an * result(size - 1);

        int j = size - 2;
        do {
          j--;
          l1 -= 1.0;

          an = impl::alpha_term_backward(l1, l2, l3, m1, m2, m3);
          beta = impl::beta_term_backward(l1, l2, l3, m1, m2, m3);

          result(j) = an * result(j + 2) + beta * result(j + 2);

          if (std::fabs(result(j)) > huge_sentinel) {
            occ::log::debug("renormalized backward recursion in wigner3j");
            result.segment(j, size - j) /= huge_sentinel;
          }

        } while (j > (i - 2));

        double lambda =
            (l1_mid_plus_one * result(j + 2) + l1_mid * result(j + 1) +
             l1_mid_minus_one * result(j)) /
            (l1_mid_plus_one * l1_mid_plus_one + l1_mid * l1_mid +
             l1_mid_minus_one * l1_mid_minus_one);

        result.head(j) *= lambda;
      }
    }
  }

  double sum = 0.0;
  for (int k = 0; k < size; k++) {
    sum += (2.0 * (l1min + k) + 1.0) * result(k) * result(k);
  }

  int s = (result(size - 1) < 0) ? -1 : 1;
  double c1 = std::pow(-1.0, l2 - l3 - m1) * s;
  result.array() *= c1 / std::sqrt(sum);

  return result;
}

double wigner3j_single(double l1, double l2, double l3, double m1, double m2,
                       double m3) {
  if (!(std::fabs(m1 + m2 + m3) < 1.0e-10 &&
        std::floor(l1 + l2 + l3) == (l1 + l2 + l3) &&
        l3 >= std::fabs(l1 - l2) && l3 <= l1 + l2 && std::fabs(m1) <= l1 &&
        std::fabs(m2) <= l2 && std::fabs(m3) <= l3))
    return 0.0;

  double l1min = std::max(std::fabs(l2 - l3), std::fabs(m1));
  int index = static_cast<int>(l1 - l1min);

  auto r = wigner3j(l2, l3, m1, m2, m3);
  return r(index);
}

} // namespace occ::sht
