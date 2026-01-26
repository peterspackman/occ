#pragma once
#include <occ/mults/sfunction_term.h>
#include <occ/dma/mult.h>
#include <occ/dma/binomial.h>

namespace occ::mults {

/**
 * @brief Builds filtered S-function term lists from multipole pairs
 *
 * Implements Orient's optimization: only include terms where Q(t1)*Q(t2) != 0
 * This typically skips 75-85% of terms for hexadecapole interactions.
 */
class SFunctionTermListBuilder {
public:
    /**
     * @param max_rank Maximum multipole rank (0-5)
     * @param tolerance Threshold below which multipole components are zero
     */
    explicit SFunctionTermListBuilder(int max_rank = 4, double tolerance = 1e-12);

    /**
     * @brief Build term list for electrostatic interaction
     *
     * Only includes terms where:
     * - abs(q1[t1]) > tolerance AND abs(q2[t2]) > tolerance
     * - l1 + l2 <= max_interaction_rank (if specified)
     *
     * @param mult1 Multipole at site A
     * @param mult2 Multipole at site B
     * @param max_interaction_rank Maximum l1+l2 to include (-1 = no limit)
     * @return Filtered list of non-zero terms
     */
    SFunctionTermList build_electrostatic_terms(
        const occ::dma::Mult& mult1,
        const occ::dma::Mult& mult2,
        int max_interaction_rank = -1) const;

    /**
     * @brief Get expected term count for full interaction (no filtering)
     * @param rank1 Max rank of multipole 1
     * @param rank2 Max rank of multipole 2
     * @return Total possible terms for this rank pair
     */
    static int get_total_term_count(int rank1, int rank2);

private:
    int m_max_rank;
    double m_tolerance;
    occ::dma::BinomialCoefficients m_binomial;

    // Check if term is non-zero
    bool should_include_term(
        double q1, double q2,
        int l1, int l2,
        int max_interaction_rank) const;
};

} // namespace occ::mults
