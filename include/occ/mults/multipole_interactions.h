#pragma once
#include <occ/mults/sfunction_evaluator.h>
#include <occ/mults/sfunction_term_builder.h>
#include <occ/dma/mult.h>
#include <vector>

namespace occ::mults {

/**
 * @brief High-level API for multipole electrostatic calculations
 *
 * Provides optimized multipole interaction calculations using:
 * - Term filtering to skip zero multipole components (75-99% reduction)
 * - Batch S-function evaluation
 * - Pre-computed coefficients
 *
 * This is the recommended API for production use.
 */
class MultipoleInteractions {
public:
    /**
     * @brief Configuration for interaction calculations
     */
    struct Config {
        int max_rank;               // Maximum multipole rank to support
        double zero_tolerance;      // Threshold for zero components
        int max_interaction_rank;   // Limit l1+l2 (-1 = no limit)

        // Default constructor with default values
        Config() : max_rank(4), zero_tolerance(1e-12), max_interaction_rank(-1) {}
    };

    MultipoleInteractions() : MultipoleInteractions(Config{}) {}
    explicit MultipoleInteractions(const Config& config);

    /**
     * @brief Compute electrostatic potential at a point
     *
     * ESP is computed as interaction energy between multipole and unit charge.
     * Uses term filtering to skip zero multipole components.
     *
     * @param multipole Multipole distribution at site
     * @param site_position Position of multipole site
     * @param eval_point Position where ESP is evaluated
     * @return ESP value in atomic units
     */
    double compute_esp(
        const occ::dma::Mult& multipole,
        const Vec3& site_position,
        const Vec3& eval_point) const;

    /**
     * @brief Compute ESP at multiple points (batch)
     *
     * More efficient than calling compute_esp() in a loop because
     * the term list is built once and reused.
     *
     * @param multipole Multipole distribution
     * @param site_position Position of multipole site
     * @param eval_points 3xN matrix where each column is an evaluation point
     * @return Vector of ESP values (N elements, same order as eval_points columns)
     */
    std::vector<double> compute_esp_grid(
        const occ::dma::Mult& multipole,
        const Vec3& site_position,
        Mat3NConstRef eval_points) const;

    /**
     * @brief Compute interaction energy between two multipoles
     *
     * Uses term filtering and batch evaluation for optimal performance.
     * Typically 3-5x faster than naive evaluation for hexadecapole interactions.
     *
     * @param mult1 First multipole distribution
     * @param pos1 Position of first multipole
     * @param mult2 Second multipole distribution
     * @param pos2 Position of second multipole
     * @return Interaction energy in atomic units
     */
    double compute_interaction_energy(
        const occ::dma::Mult& mult1, const Vec3& pos1,
        const occ::dma::Mult& mult2, const Vec3& pos2) const;

    // Accessors
    const Config& config() const { return m_config; }
    const SFunctionEvaluator& evaluator() const { return m_evaluator; }

private:
    Config m_config;
    SFunctionTermListBuilder m_builder;
    mutable SFunctionEvaluator m_evaluator;  // mutable for set_coordinates in const methods

    // Helper: accumulate energy from S-function results
    static double accumulate_energy(
        const SFunctionTermList& terms,
        const std::vector<SFunctionResult>& results,
        double r);
};

} // namespace occ::mults
