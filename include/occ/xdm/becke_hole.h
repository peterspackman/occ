#pragma once

namespace occ::xdm {

double becke_hole_br89_analytic(double rho, double Q, double norm);
double becke_hole_br89_newton(double rho, double Q, double norm);

inline double becke_hole_br89(double rho, double Q, double norm,
                              bool analytic = true) {
    if (analytic)
        return occ::xdm::becke_hole_br89_analytic(rho, Q, norm);
    else
        return occ::xdm::becke_hole_br89_newton(rho, Q, norm);
}

} // namespace occ::xdm
