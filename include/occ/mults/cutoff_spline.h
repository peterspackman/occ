#pragma once
#include <cmath>

namespace occ::mults {

/// DMACRYS-style radial spline taper definition.
/// f(r)=1 for r<=r_on, spline in (r_on,r_off], f(r)=0 for r>r_off.
struct CutoffSpline {
    bool enabled = false;
    double r_on = 0.0;   // Angstrom
    double r_off = 0.0;  // Angstrom
    int order = 3;       // 3 (cubic) or 5 (quintic)

    bool is_valid() const {
        return enabled && (r_off > r_on) && (order == 3 || order == 5);
    }
};

struct CutoffSplineValue {
    double value = 1.0;              // f(r)
    double first_derivative = 0.0;   // df/dr (Angstrom^-1)
    double second_derivative = 0.0;  // d2f/dr2 (Angstrom^-2)
};

inline CutoffSplineValue evaluate_cutoff_spline(
    double r,
    double r_on,
    double r_off,
    int order = 3) {

    CutoffSplineValue out;
    if (r_off <= r_on) {
        out.value = (r <= r_on) ? 1.0 : 0.0;
        return out;
    }
    if (r <= r_on) {
        return out;
    }
    if (r > r_off) {
        out.value = 0.0;
        return out;
    }

    const double vdr = 1.0 / (r_on - r_off);
    const double vdr2 = vdr * vdr;
    const double vdr3 = vdr2 * vdr;

    const double dr = r - r_off;
    const double dr2 = dr * dr;
    const double dr3 = dr2 * dr;

    if (order == 3) {
        const double c2 = 3.0 * vdr2;
        const double c3 = -2.0 * vdr3;
        out.value = c2 * dr2 + c3 * dr3;
        out.first_derivative = 2.0 * c2 * dr + 3.0 * c3 * dr2;
        out.second_derivative = 2.0 * c2 + 6.0 * c3 * dr;
        return out;
    }

    // Quintic spline (C2 continuous): DMACRYS order=5.
    const double vdr4 = vdr3 * vdr;
    const double vdr5 = vdr4 * vdr;
    const double dr4 = dr3 * dr;
    const double dr5 = dr4 * dr;

    const double c3 = -8.0 * vdr3;
    const double c4 = 21.0 * vdr4;
    const double c5 = -12.0 * vdr5;
    out.value = c3 * dr3 + c4 * dr4 + c5 * dr5;
    out.first_derivative = 3.0 * c3 * dr2 + 4.0 * c4 * dr3 + 5.0 * c5 * dr4;
    out.second_derivative = 6.0 * c3 * dr + 12.0 * c4 * dr2 + 20.0 * c5 * dr3;
    return out;
}

inline CutoffSplineValue evaluate_cutoff_spline(
    double r,
    const CutoffSpline& spline) {
    if (!spline.is_valid()) {
        return {};
    }
    return evaluate_cutoff_spline(r, spline.r_on, spline.r_off, spline.order);
}

} // namespace occ::mults
