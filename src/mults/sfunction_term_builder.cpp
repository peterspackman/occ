#include <occ/mults/sfunction_term_builder.h>
#include <occ/mults/sfunction_evaluator.h>
#include <cmath>

namespace occ::mults {

SFunctionTermListBuilder::SFunctionTermListBuilder(int max_rank, double tolerance)
    : m_max_rank(max_rank),
      m_tolerance(tolerance),
      m_binomial(max_rank + 4)
{
}

SFunctionTermList SFunctionTermListBuilder::build_electrostatic_terms(
    const occ::dma::Mult& mult1,
    const occ::dma::Mult& mult2,
    int max_interaction_rank) const
{
    SFunctionTermList result;

    // Determine effective ranks
    int rank1 = mult1.max_rank;
    int rank2 = mult2.max_rank;

    if (max_interaction_rank < 0) {
        max_interaction_rank = rank1 + rank2;
    }

    // Reserve space (estimate: ~30% of possible terms survive filtering)
    int total_possible = get_total_term_count(rank1, rank2);
    result.reserve(total_possible / 3);

    // Loop over all multipole component pairs
    for (int t1 = 0; t1 < mult1.num_components(); ++t1) {
        double q1 = mult1.q(t1);

        // Skip if q1 is essentially zero
        if (std::abs(q1) < m_tolerance) continue;

        for (int t2 = 0; t2 < mult2.num_components(); ++t2) {
            double q2 = mult2.q(t2);

            // Skip if q2 is essentially zero
            if (std::abs(q2) < m_tolerance) continue;

            // Get (l, m) quantum numbers
            auto [l1, m1] = SFunctionEvaluator::index_to_lm(t1);
            auto [l2, m2] = SFunctionEvaluator::index_to_lm(t2);

            // j = l1 + l2 for electrostatic interactions (Orient convention)
            int j = l1 + l2;

            // Skip if exceeds max interaction rank
            if (j > max_interaction_rank) continue;

            // Compute coefficient: q1 * q2 * binomial(j, l1)
            double fac = m_binomial.binomial(j, l1);
            double coeff = q1 * q2 * fac;

            // Power of R: (l1 + l2 + 1) for electrostatic
            int power = l1 + l2 + 1;

            // Add term to list
            result.add_term(t1, t2, j, coeff, power);
        }
    }

    return result;
}

int SFunctionTermListBuilder::get_total_term_count(int rank1, int rank2) {
    // Number of components for each rank: (rank+1)^2
    int n1 = (rank1 + 1) * (rank1 + 1);
    int n2 = (rank2 + 1) * (rank2 + 1);
    return n1 * n2;
}

bool SFunctionTermListBuilder::should_include_term(
    double q1, double q2,
    int l1, int l2,
    int max_interaction_rank) const
{
    // Check magnitudes
    if (std::abs(q1) < m_tolerance || std::abs(q2) < m_tolerance) {
        return false;
    }

    // Check rank limit
    if (max_interaction_rank >= 0 && (l1 + l2) > max_interaction_rank) {
        return false;
    }

    return true;
}

} // namespace occ::mults
