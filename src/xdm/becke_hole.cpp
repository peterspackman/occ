#include <cmath>
#include <occ/xdm/becke_hole.h>

namespace occ::xdm {

void xfuncs(double x, double rhs, double &f, double &df) {
  // working
  double expo23 = std::exp(-2.0 / 3.0 * x);
  f = x * expo23 / (x - 2.0) - rhs;
  df = 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / ((x - 2.0) * (x - 2.0)) * expo23;
}

double becke_hole_br89_analytic(double rho, double Q, double norm) {
  constexpr double alpha1 = 1.5255251812009530;
  constexpr double alpha2 = 0.4576575543602858;
  constexpr double alpha3 = 0.4292036732051034;

  constexpr double c[6] = {0.7566445420735584, -2.6363977871370960,
                           5.4745159964232880, -12.657308127108290,
                           4.1250584725121360, -30.42513395716384};

  constexpr double b[6] = {0.4771976183772063, -1.7799813494556270,
                           3.8433841862302150, -9.5912050880518490,
                           2.1730180285916720, -30.425133851603660};

  constexpr double d[6] = {0.00004435009886795587, 0.58128653604457910,
                           66.742764515940610,     434.26780897229770,
                           824.7765766052239000,   1657.9652731582120};
  constexpr double e[6] = {0.00003347285060926091, 0.47917931023971350,
                           62.392268338574240,     463.14816427938120,
                           785.2360350104029000,   1657.962968223273000000};
  constexpr double B = 2.085749716493756;

  constexpr double third2 = 2.0 / 3.0;
  const double y = third2 * std::pow(M_PI * rho / norm, third2) * rho / Q;
  double x = 0.0;
  if (y <= 0) {
    const double g = -std::atan(alpha1 * y + alpha2) + alpha3;
    double p1y = 0.0;
    double p2y = 0.0;
    double yi = 1.0;
    for (int i = 0; i < 6; i++) {
      p1y += c[i] * yi;
      p2y += b[i] * yi;
      yi *= y;
    }
    x = g * p1y / p2y;
  } else {
    const double By = B * y;
    // inverse hyperbolic cosecant
    const double g = std::log(1.0 / By + std::sqrt(1.0 / (By * By) + 1)) + 2;
    double p1y = 0.0;
    double p2y = 0.0;
    double yi = 1.0;
    for (int i = 0; i < 6; i++) {
      p1y += d[i] * yi;
      p2y += e[i] * yi;
      yi *= y;
    }
    x = g * p1y / p2y;
  }
  const double expo = std::exp(-x);
  const double prefac = rho / expo;
  const double alf = std::pow(8.0 * M_PI * prefac / norm, 1.0 / 3.0);
  return x / alf;
}

double becke_hole_br89_newton(double rho, double quad, double hnorm) {
  double x1{0}, f{0}, df{0};
  double third2 = 2.0 / 3.0;
  double x = 0.0;
  const double rhs = third2 * std::pow(M_PI * rho / hnorm, third2) * rho / quad;
  double x0 = 2.0;
  double shift = (rhs > 0) ? 1.0 : -1.0;
  bool initialized = false;
  for (int i = 0; i < 16; i++) {
    x = x0 + shift;
    xfuncs(x, rhs, f, df);
    if (f * rhs > 0.0) {
      initialized = true;
      break;
    }
    shift = 0.1 * shift;
  }
  if (!initialized)
    throw "Failed to initialize Newton's method for Becke hole";

  bool converged = false;
  for (int i = 0; i < 100; i++) {
    xfuncs(x, rhs, f, df);
    x1 = x - f / df;
    if (std::abs(x1 - x) < 1e-10) {
      converged = true;
      break;
    }
    x = x1;
  }
  if (!converged)
    throw "Failed to converge Newton's method for Becke hole";
  x = x1;
  const double expo = std::exp(-x);
  const double prefac = rho / expo;
  const double alf = std::pow(8.0 * M_PI * prefac / hnorm, 1.0 / 3.0);
  return x / alf;
}

} // namespace occ::xdm
