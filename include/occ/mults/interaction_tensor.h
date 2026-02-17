#pragma once
#include <occ/ints/rints.h>
#include <array>
#include <cmath>

namespace occ::mults {

using occ::ints::nherm;
using occ::ints::nhermsum;
using occ::ints::hermite_index;

/// Cartesian interaction tensor T_{tuv} = d^{t+u+v}/dx^t dy^u dz^v (1/R)
///
/// Uses the same index layout as RInts (hermite_index) with point-charge
/// base cases instead of Boys functions.
///
/// @tparam MaxL Maximum total rank (t+u+v). For hex-hex interactions, MaxL=8.
template <int MaxL>
struct InteractionTensor {
    static constexpr int size = nhermsum(MaxL);
    alignas(64) double data[size];

    double &operator()(int t, int u, int v) {
        return data[hermite_index(t, u, v)];
    }

    double operator()(int t, int u, int v) const {
        return data[hermite_index(t, u, v)];
    }
};

/// Compute Cartesian interaction tensor via Obara-Saika-style recurrence.
///
/// Base case: T^{(m)}_{000} = (-1)^m (2m-1)!! / R^{2m+1}
///
/// Recurrence (same as compute_r_ints in rints.h):
///   T^{(m)}_{t+1,u,v} = Rx * T^{(m+1)}_{t,u,v} + t * T^{(m+1)}_{t-1,u,v}
///   (analogous for u,v directions)
///
/// Result: T_{tuv} = T^{(0)}_{tuv}
///
/// @param Rx  x-component of displacement R = pos2 - pos1
/// @param Ry  y-component of displacement
/// @param Rz  z-component of displacement
/// @param T   Output tensor
template <int MaxL>
void compute_interaction_tensor(double Rx, double Ry, double Rz,
                                InteractionTensor<MaxL> &T) {
    const double R2 = Rx * Rx + Ry * Ry + Rz * Rz;
    const double R = std::sqrt(R2);
    const double invR = 1.0 / R;
    const double invR2 = invR * invR;

    constexpr int nsz = nhermsum(MaxL);
    constexpr int num_aux = MaxL + 2;
    double R_all[num_aux * nsz];

    auto R_m = [&](int m, int t, int u, int v) -> double & {
        return R_all[m * nsz + hermite_index(t, u, v)];
    };

    // Base case: T^{(m)}_{000} = (-1)^m (2m-1)!! / R^{2m+1}
    // Recurrence: T^{(m)}_{000} = -(2m-1) * invR2 * T^{(m-1)}_{000}
    R_m(0, 0, 0, 0) = invR;
    for (int m = 1; m <= MaxL; ++m) {
        R_m(m, 0, 0, 0) = -(2 * m - 1) * invR2 * R_m(m - 1, 0, 0, 0);
    }

    // Build up t index (x-direction)
    for (int t = 0; t < MaxL; ++t) {
        for (int m = 0; m <= MaxL - t - 1; ++m) {
            double val = Rx * R_m(m + 1, t, 0, 0);
            if (t > 0) {
                val += t * R_m(m + 1, t - 1, 0, 0);
            }
            R_m(m, t + 1, 0, 0) = val;
        }
    }

    // Build up u index (y-direction)
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u < MaxL - t; ++u) {
            for (int m = 0; m <= MaxL - t - u - 1; ++m) {
                double val = Ry * R_m(m + 1, t, u, 0);
                if (u > 0) {
                    val += u * R_m(m + 1, t, u - 1, 0);
                }
                R_m(m, t, u + 1, 0) = val;
            }
        }
    }

    // Build up v index (z-direction)
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v < MaxL - t - u; ++v) {
                for (int m = 0; m <= MaxL - t - u - v - 1; ++m) {
                    double val = Rz * R_m(m + 1, t, u, v);
                    if (v > 0) {
                        val += v * R_m(m + 1, t, u, v - 1);
                    }
                    R_m(m, t, u, v + 1) = val;
                }
            }
        }
    }

    // Extract m=0 results
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v <= MaxL - t - u; ++v) {
                T(t, u, v) = R_m(0, t, u, v);
            }
        }
    }
}

} // namespace occ::mults
