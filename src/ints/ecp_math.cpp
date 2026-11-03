// Special functions and tables for the native ECP integrals: scaled modified
// spherical Bessel functions, Boys and G_m functions, real harmonic
// polynomials, unit-sphere monomial integrals and Gauss-Legendre nodes.
#include "ecp_internal.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace occ::ints::ecp_detail {

namespace {

// e^{-z} z^l sum_k (z^2/2)^k / (k! (2l+2k+1)!!); all terms positive.
double scaled_bessel_series(int l, double z) {
  double t = std::exp(-z);
  for (int i = 1; i <= l; i++)
    t *= z / (2 * i + 1);
  double s = t;
  const double h = 0.5 * z * z;
  for (int k = 1; k < 100000; k++) {
    t *= h / (k * (2.0 * l + 2.0 * k + 1.0));
    const double next = s + t;
    if (next == s)
      break;
    s = next;
  }
  return s;
}

} // namespace

void scaled_bessel_i(int lmax, double z, double *out) {
  if (z <= 0.0) {
    out[0] = 1.0;
    for (int l = 1; l <= lmax; l++)
      out[l] = 0.0;
    return;
  }
  // Upward recurrence from the closed forms is stable once z is well above
  // lmax^2 / 2; below that, start from the series at the top two orders and
  // recur downwards.
  const double upward_threshold = std::max(2.0, 0.5 * lmax * lmax);
  if (z > upward_threshold) {
    const double e = std::exp(-2.0 * z);
    out[0] = -std::expm1(-2.0 * z) / (2.0 * z);
    if (lmax >= 1)
      out[1] = (1.0 + e) / (2.0 * z) - out[0] / z;
    const double inv_z = 1.0 / z;
    for (int l = 1; l < lmax; l++)
      out[l + 1] = out[l - 1] - (2 * l + 1) * inv_z * out[l];
    return;
  }
  if (z < 1.0 || lmax < 2) {
    for (int l = 0; l <= lmax; l++)
      out[l] = scaled_bessel_series(l, z);
    return;
  }
  out[lmax] = scaled_bessel_series(lmax, z);
  out[lmax - 1] = scaled_bessel_series(lmax - 1, z);
  const double inv_z = 1.0 / z;
  for (int l = lmax - 1; l > 0; l--)
    out[l - 1] = out[l + 1] + (2 * l + 1) * inv_z * out[l];
}

void boys(int mmax, double T, double *out) {
  const double threshold = std::max(35.0, mmax + 15.0);
  if (T > threshold) {
    const double e = std::exp(-T);
    const double sT = std::sqrt(T);
    out[0] = 0.5 * std::sqrt(std::numbers::pi / T) * std::erf(sT);
    const double inv2T = 0.5 / T;
    for (int m = 0; m < mmax; m++)
      out[m + 1] = ((2 * m + 1) * out[m] - e) * inv2T;
    return;
  }
  // F_mmax by its positive series, then downward recursion.
  const double e = std::exp(-T);
  double t = 1.0 / (2 * mmax + 1);
  double s = t;
  for (int k = 1; k < 100000; k++) {
    t *= 2.0 * T / (2 * mmax + 2 * k + 1);
    const double next = s + t;
    if (next == s)
      break;
    s = next;
  }
  out[mmax] = e * s;
  for (int m = mmax; m > 0; m--)
    out[m - 1] = (2.0 * T * out[m] + e) / (2 * m - 1);
}

void gfunc(int mmax, double T, double *out) {
  for (int m = 0; m <= mmax; m++) {
    if (T <= std::max(60.0, 3.0 * m + 30.0)) {
      // e^{-T} sum_k T^k/k! B(k+1/2, m+1)/2
      double t = std::exp(-T + std::lgamma(0.5) + std::lgamma(m + 1.0) -
                          std::lgamma(m + 1.5)) *
                 0.5;
      double s = t;
      for (int k = 0; k < 100000; k++) {
        t *= T / (k + 1) * (k + 0.5) / (k + m + 1.5);
        const double next = s + t;
        if (next == s && k > T)
          break;
        s = next;
      }
      out[m] = s;
    } else {
      // (1/2) sum_k (2k-1)!!/(2k)!! (m+k)! / T^(m+k+1)
      double s = 0.0, c = 1.0, prev = 1e300;
      const double lnT = std::log(T);
      for (int k = 0; k < 1000; k++) {
        const double term =
            0.5 * c * std::exp(std::lgamma(m + k + 1.0) - (m + k + 1) * lnT);
        if (term > prev || s + term == s)
          break;
        s += term;
        prev = term;
        c *= (2.0 * k + 1.0) / (2.0 * k + 2.0);
      }
      out[m] = s;
    }
  }
}

double sphere_monomial_integral(int a, int b, int c) {
  if ((a | b | c) & 1)
    return 0.0;
  // 4 pi (a-1)!! (b-1)!! (c-1)!! / (a+b+c+1)!!, built as a product of ratios
  double v = 4.0 * std::numbers::pi;
  int num_a = a - 1, num_b = b - 1, num_c = c - 1;
  int den = a + b + c + 1;
  // Pair each numerator factor with a denominator factor to stay in range.
  while (num_a > 0 || num_b > 0 || num_c > 0 || den > 1) {
    double f = 1.0;
    if (num_a > 0) {
      f *= num_a;
      num_a -= 2;
    } else if (num_b > 0) {
      f *= num_b;
      num_b -= 2;
    } else if (num_c > 0) {
      f *= num_c;
      num_c -= 2;
    }
    if (den > 1) {
      f /= den;
      den -= 2;
    }
    v *= f;
  }
  return v;
}

namespace {

using Poly = std::vector<double>; // homogeneous polynomial of degree d

Poly poly_mul(const Poly &p, int dp, const Poly &q, int dq) {
  const int d = dp + dq;
  Poly r(nmono(d), 0.0);
  for (int i1 = dp, a = 0; i1 >= 0; i1--) {
    for (int j1 = dp - i1; j1 >= 0; j1--, a++) {
      if (p[a] == 0.0)
        continue;
      const int k1 = dp - i1 - j1;
      for (int i2 = dq, b = 0; i2 >= 0; i2--) {
        for (int j2 = dq - i2; j2 >= 0; j2--, b++) {
          const int k2 = dq - i2 - j2;
          r[ecp_cart_index(i1 + i2, j1 + j2, k1 + k2)] += p[a] * q[b];
        }
      }
    }
  }
  return r;
}

double poly_sphere_norm2(const Poly &p, int d) {
  double s = 0.0;
  for (int i1 = d, a = 0; i1 >= 0; i1--) {
    for (int j1 = d - i1; j1 >= 0; j1--, a++) {
      if (p[a] == 0.0)
        continue;
      const int k1 = d - i1 - j1;
      for (int i2 = d, b = 0; i2 >= 0; i2--) {
        for (int j2 = d - i2; j2 >= 0; j2--, b++) {
          const int k2 = d - i2 - j2;
          s += p[a] * p[b] *
               sphere_monomial_integral(i1 + i2, j1 + j2, k1 + k2);
        }
      }
    }
  }
  return s;
}

double binomial(int n, int k) {
  if (k < 0 || k > n)
    return 0.0;
  double r = 1.0;
  for (int i = 1; i <= k; i++)
    r = r * (n - k + i) / i;
  return r;
}

// Real harmonic polynomials of degree l, orthonormal on the unit sphere.
// Rows are m = -l..l (the order is irrelevant to the ECP: only sums over m
// appear). Built as Re/Im (x+iy)^|m| times d^|m| P_l(z) homogenised with r^2.
Mat real_harmonics(int l) {
  Mat Y = Mat::Zero(2 * l + 1, nmono(l));
  // Legendre P_l(z) = 2^-l sum_k (-1)^k C(l,k) C(2l-2k,l) z^(l-2k)
  std::vector<double> P(l + 1, 0.0);
  for (int k = 0; 2 * k <= l; k++)
    P[l - 2 * k] = std::pow(-1.0, k) * binomial(l, k) *
                   binomial(2 * l - 2 * k, l) / std::pow(2.0, l);
  Poly r2(nmono(2), 0.0);
  r2[ecp_cart_index(2, 0, 0)] = 1.0;
  r2[ecp_cart_index(0, 2, 0)] = 1.0;
  r2[ecp_cart_index(0, 0, 2)] = 1.0;
  for (int m = -l; m <= l; m++) {
    const int am = std::abs(m);
    // Q(z) = d^am P_l / dz^am
    std::vector<double> Q(l - am + 1, 0.0);
    for (int c = 0; c + am <= l; c++) {
      double f = 1.0;
      for (int t = 0; t < am; t++)
        f *= (c + am - t);
      Q[c] = P[c + am] * f;
    }
    const int dq = l - am;
    Poly H(nmono(dq), 0.0);
    for (int c = 0; c <= dq; c++) {
      if (Q[c] == 0.0 || ((dq - c) & 1))
        continue;
      Poly term(nmono(c), 0.0);
      term[ecp_cart_index(0, 0, c)] = Q[c];
      int dt = c;
      for (int k = 0; k < (dq - c) / 2; k++) {
        term = poly_mul(term, dt, r2, 2);
        dt += 2;
      }
      for (size_t i = 0; i < H.size(); i++)
        H[i] += term[i];
    }
    Poly phi(nmono(am), 0.0);
    for (int s = 0; s <= am; s++) {
      const double b = binomial(am, s);
      if (m >= 0 && (s % 2 == 0))
        phi[ecp_cart_index(am - s, s, 0)] = b * ((s / 2) % 2 ? -1.0 : 1.0);
      if (m < 0 && (s % 2 == 1))
        phi[ecp_cart_index(am - s, s, 0)] =
            b * (((s - 1) / 2) % 2 ? -1.0 : 1.0);
    }
    Poly y = poly_mul(phi, am, H, dq);
    const double n2 = poly_sphere_norm2(y, l);
    const double inv = 1.0 / std::sqrt(n2);
    for (int i = 0; i < nmono(l); i++)
      Y(m + l, i) = y[i] * inv;
  }
  return Y;
}

void gauss_legendre(int n, std::vector<double> &x, std::vector<double> &w) {
  x.resize(n);
  w.resize(n);
  // P_n(z) and P_n'(z) by the three-term recurrence
  auto legendre = [n](double z, double &p, double &dp) {
    double p1 = 1.0, p2 = 0.0;
    for (int j = 1; j <= n; j++) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
    }
    p = p1;
    dp = n * (z * p1 - p2) / (z * z - 1.0);
  };
  for (int i = 0; i < (n + 1) / 2; i++) {
    double z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
    double p, dp;
    for (int it = 0; it < 100; it++) {
      legendre(z, p, dp);
      const double z1 = z;
      z = z1 - p / dp;
      if (std::abs(z - z1) < 1e-16)
        break;
    }
    legendre(z, p, dp);
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
}

} // namespace
} // namespace occ::ints::ecp_detail

namespace occ::ints {
using namespace ecp_detail;

EcpEvaluator::Tables::Tables(int max_shell_l, int max_semilocal_l)
    : max_shell_l(max_shell_l), max_semilocal_l(max_semilocal_l) {
  max_lambda = max_shell_l + std::max(max_semilocal_l, 0);
  max_monomial_degree = max_shell_l + max_lambda;
  Y.resize(max_lambda + 1);
  for (int l = 0; l <= max_lambda; l++)
    Y[l] = real_harmonics(l);
  // S[l](m, global monomial) = int r^M Y_lm dOmega
  S.resize(std::max(max_semilocal_l, 0) + 1);
  const int nm = nmono_cumulative(max_monomial_degree);
  for (int l = 0; l <= max_semilocal_l; l++) {
    S[l] = Mat::Zero(2 * l + 1, nm);
    for (int d = 0; d <= max_monomial_degree; d++) {
      if (d < l || ((d - l) & 1))
        continue;
      for (int i = d; i >= 0; i--) {
        for (int j = d - i; j >= 0; j--) {
          const int k = d - i - j;
          const int col = mono_offset(d) + ecp_cart_index(i, j, k);
          for (int m = 0; m < 2 * l + 1; m++) {
            double s = 0.0;
            for (int i2 = l, b = 0; i2 >= 0; i2--) {
              for (int j2 = l - i2; j2 >= 0; j2--, b++) {
                const double y = Y[l](m, b);
                if (y == 0.0)
                  continue;
                s += y * sphere_monomial_integral(i + i2, j + j2,
                                                  k + (l - i2 - j2));
              }
            }
            S[l](m, col) = s;
          }
        }
      }
    }
  }
  gauss_legendre(num_radial_points, leg_x, leg_w);
  binom.resize(max_shell_l + 1);
  for (int n = 0; n <= max_shell_l; n++) {
    binom[n].resize(n + 1);
    for (int k = 0; k <= n; k++)
      binom[n][k] = binomial(n, k);
  }
}

} // namespace occ::ints
