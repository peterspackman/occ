#pragma once
#include <vector>

namespace occ::mults {

/**
 * @brief Represents a single S-function term to evaluate
 *
 * Corresponds to one entry in Orient's list(:), q1(:), q2(:), fac(:) arrays
 */
struct SFunctionTerm {
    int t1;        // Multipole index for site A
    int t2;        // Multipole index for site B
    int j;         // Combined rank parameter
    double coeff;  // Pre-computed q1 * q2 * binomial_coefficient
    int power;     // R^-power for interaction energy

    SFunctionTerm() = default;

    SFunctionTerm(int t1_, int t2_, int j_, double coeff_, int power_)
        : t1(t1_), t2(t2_), j(j_), coeff(coeff_), power(power_) {}
};

/**
 * @brief List of S-function terms for a multipole pair interaction
 *
 * Analogous to Orient's filtered term arrays
 */
struct SFunctionTermList {
    std::vector<SFunctionTerm> terms;

    size_t size() const { return terms.size(); }
    bool empty() const { return terms.empty(); }

    // Reserve space for expected number of terms
    void reserve(size_t n) { terms.reserve(n); }

    // Add a term to the list
    void add_term(int t1, int t2, int j, double coeff, int power) {
        terms.emplace_back(t1, t2, j, coeff, power);
    }
};

} // namespace occ::mults
