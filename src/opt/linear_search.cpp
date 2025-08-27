/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <occ/opt/linear_search.h>
#include <occ/core/log.h>
#include <cmath>
#include <limits>
#include <array>

namespace occ::opt {

std::pair<double, double> fit_cubic(double y0, double y1, double g0, double g1) {
    // Cubic polynomial: ax^3 + bx^2 + cx + d
    // Constraints:
    //   f(0) = y0 => d = y0
    //   f(1) = y1 => a + b + c + d = y1
    //   f'(0) = g0 => c = g0
    //   f'(1) = g1 => 3a + 2b + c = g1
    
    double d = y0;
    double c = g0;
    // From f(1) = y1: a + b = y1 - c - d = y1 - g0 - y0
    // From f'(1) = g1: 3a + 2b = g1 - c = g1 - g0
    // Solving: a = g0 + g1 - 2(y1 - y0)
    //          b = -2g0 - g1 + 3(y1 - y0)
    
    double a = g0 + g1 - 2.0 * (y1 - y0);
    double b = -2.0 * g0 - g1 + 3.0 * (y1 - y0);
    
    // Find critical points: 3ax^2 + 2bx + c = 0
    double discriminant = 4.0 * b * b - 12.0 * a * c;
    
    if (discriminant < 0) {
        // No real critical points
        return {std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::quiet_NaN()};
    }
    
    double sqrt_disc = std::sqrt(discriminant);
    double r1 = (-2.0 * b + sqrt_disc) / (6.0 * a);
    double r2 = (-2.0 * b - sqrt_disc) / (6.0 * a);
    
    double minim, maxim;
    if (a > 0) {
        maxim = std::min(r1, r2);
        minim = std::max(r1, r2);
    } else {
        minim = std::min(r1, r2);
        maxim = std::max(r1, r2);
    }
    
    // Check conditions for valid minimum
    if (0 < maxim && maxim < 1 && std::abs(minim - 0.5) > std::abs(maxim - 0.5)) {
        return {std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::quiet_NaN()};
    }
    
    // Evaluate polynomial at minimum
    double min_val = a * minim * minim * minim + b * minim * minim + c * minim + d;
    return {minim, min_val};
}

std::pair<double, double> fit_quartic(double y0, double y1, double g0, double g1) {
    // Helper lambda to construct quartic coefficients
    auto g_func = [](double y0, double y1, double g0, double g1, double c) -> std::array<double, 5> {
        double a = c + 3.0 * (y0 - y1) + 2.0 * g0 + g1;
        double b = -2.0 * c - 4.0 * (y0 - y1) - 3.0 * g0 - g1;
        return {a, b, c, g0, y0};  // coefficients for ax^4 + bx^3 + cx^2 + dx + e
    };
    
    // Helper lambda to find minimum of quartic polynomial
    auto quart_min = [](const std::array<double, 5>& p) -> std::pair<double, double> {
        // Find roots of derivative: 4ax^3 + 3bx^2 + 2cx + d = 0
        double a = 4.0 * p[0];
        double b = 3.0 * p[1]; 
        double c = 2.0 * p[2];
        double d = p[3];
        
        // Use Newton-Raphson to find critical point
        double minim = 0.5;  // Initial guess
        
        for (int i = 0; i < 20; i++) {
            double f = a * minim * minim * minim + b * minim * minim + c * minim + d;
            double df = 3.0 * a * minim * minim + 2.0 * b * minim + c;
            if (std::abs(df) < 1e-12) break;
            minim -= f / df;
        }
        
        // Evaluate polynomial at minimum
        double minval = p[0] * std::pow(minim, 4) + p[1] * std::pow(minim, 3) + 
                       p[2] * minim * minim + p[3] * minim + p[4];
        return {minim, minval};
    };
    
    // Discriminant of d^2y/dx^2=0
    double D = -((g0 + g1) * (g0 + g1)) - 2.0 * g0 * g1 + 
               6.0 * (y1 - y0) * (g0 + g1) - 6.0 * (y1 - y0) * (y1 - y0);
    
    if (D < 1e-11) {
        return {std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::quiet_NaN()};
    }
    
    double m = -5.0 * g0 - g1 - 6.0 * y0 + 6.0 * y1;
    auto p1 = g_func(y0, y1, g0, g1, 0.5 * (m + std::sqrt(2.0 * D)));
    auto p2 = g_func(y0, y1, g0, g1, 0.5 * (m - std::sqrt(2.0 * D)));
    
    if (p1[0] < 0 && p2[0] < 0) {
        return {std::numeric_limits<double>::quiet_NaN(), 
                std::numeric_limits<double>::quiet_NaN()};
    }
    
    auto [minim1, minval1] = quart_min(p1);
    auto [minim2, minval2] = quart_min(p2);
    
    if (minval1 < minval2) {
        return {minim1, minval1};
    } else {
        return {minim2, minval2};
    }
}

std::pair<double, double> linear_search(double E0, double E1, double g0, double g1) {
    occ::log::trace("Linear interpolation: E0={:.8f}, E1={:.8f}, g0={:.6f}, g1={:.6f}", 
                    E0, E1, g0, g1);
    
    // Try quartic fit first
    auto [t, E] = fit_quartic(E0, E1, g0, g1);
    bool quartic_success = !std::isnan(t) && t >= -1.0 && t <= 2.0;
    
    if (!quartic_success) {
        // Try cubic fit
        std::tie(t, E) = fit_cubic(E0, E1, g0, g1);
        bool cubic_success = !std::isnan(t) && t >= 0.0 && t <= 1.0;
        
        if (!cubic_success) {
            // No fit succeeded - use simple selection
            if (E0 <= E1) {
                occ::log::debug("No fit succeeded, staying in new point");
                return {0.0, E0};
            } else {
                occ::log::debug("No fit succeeded, returning to best point");
                return {1.0, E1};
            }
        } else {
            occ::log::debug("Cubic interpolation: t = {:.6f}", t);
        }
    } else {
        occ::log::debug("Quartic interpolation: t = {:.6f}", t);
    }
    
    occ::log::trace("Interpolated energy: {:.8f}", E);
    return {t, E};
}

}  // namespace occ::opt