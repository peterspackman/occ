#pragma once
#include <occ/mults/sfunction_result.h>
#include <occ/mults/sfunction_term.h>
#include <occ/mults/coordinate_system.h>
#include <occ/dma/binomial.h>
#include <vector>

namespace occ::mults {

/**
 * @brief S-function evaluator with batch evaluation support
 *
 * Wraps existing S-function logic and adds:
 * - Batch evaluation for multiple terms
 * - Coordinate system caching
 * - Binomial coefficient caching
 */
class SFunctionEvaluator {
public:
    explicit SFunctionEvaluator(int max_rank = 4);

    /**
     * @brief Set coordinate system for subsequent evaluations
     * @param ra Position of site A
     * @param rb Position of site B
     */
    void set_coordinates(const Vec3& ra, const Vec3& rb);

    /**
     * @brief Set coordinate system directly (for body-frame S-functions)
     * @param coords Pre-computed coordinate system with body-frame unit vectors
     */
    void set_coordinate_system(const CoordinateSystem& coords);

    /**
     * @brief Compute single S-function
     * @param t1 Multipole index for site A
     * @param t2 Multipole index for site B
     * @param j Combined rank parameter
     * @param deriv_level Derivative level (0=value only, 1=first derivs)
     * @return S-function result
     */
    SFunctionResult compute(int t1, int t2, int j, int deriv_level = 0);

    /**
     * @brief Batch evaluate S-functions from term list
     * @param term_list Pre-filtered list of terms
     * @param deriv_level Derivative level
     * @return Vector of results (same size and order as term_list.terms)
     */
    std::vector<SFunctionResult> compute_batch(
        const SFunctionTermList& term_list,
        int deriv_level = 0);

    // Accessors
    const CoordinateSystem& coordinates() const { return m_coords; }
    double r() const { return m_coords.r; }
    int max_rank() const { return m_max_rank; }

    // Index mapping helpers
    static std::pair<int, int> index_to_lm(int index);
    static int lm_to_index(int l, int m);

private:
    int m_max_rank;
    CoordinateSystem m_coords;
    occ::dma::BinomialCoefficients m_binomial;

    // Internal: dispatch to appropriate computation method
    SFunctionResult compute_internal(int t1, int t2, int j, int deriv_level);
};

} // namespace occ::mults
