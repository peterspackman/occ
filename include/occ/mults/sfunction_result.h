#pragma once
#include <occ/core/linear_algebra.h>
#include <array>

namespace occ::mults {

/**
 * @brief Result of S-function evaluation including derivatives
 *
 * Storage layout matches Orient's convention:
 * - s0: Function value
 * - s1: First derivatives [15 elements]
 *   Indices 0-2: d/d(e1r_x, e1r_y, e1r_z) - unit vector at site A
 *   Indices 3-5: d/d(e2r_x, e2r_y, e2r_z) - unit vector at site B
 *   Indices 6-14: d/d(orientation matrix elements) - 9 elements for relative orientation
 * - s2: Second derivatives [not implemented yet]
 */
struct SFunctionResult {
    double s0 = 0.0;                    // Function value
    std::array<double, 15> s1{};        // First derivatives (unit vectors + orientation)
    // s2 will be added later when we implement forces/hessians

    SFunctionResult() = default;

    // Apply scaling factor (used for binomial coefficients)
    void apply_factor(double factor) {
        s0 *= factor;
        for (auto& d : s1) d *= factor;
    }

    // Accumulate another result (for summing contributions)
    SFunctionResult& operator+=(const SFunctionResult& other) {
        s0 += other.s0;
        for (size_t i = 0; i < s1.size(); ++i) s1[i] += other.s1[i];
        return *this;
    }
};

} // namespace occ::mults
