#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/sfunctions.h>
#include <vector>

namespace occ::mults {

/**
 * Electrostatic Potential (ESP) evaluation using multipoles and S-functions
 * 
 * This class evaluates the electrostatic potential at points in space due to
 * multipole distributions, following Orient's implementation.
 */
class MultipoleESP {
public:
    /**
     * Constructor
     * @param max_rank Maximum multipole rank to handle
     */
    explicit MultipoleESP(int max_rank = 4);
    
    /**
     * Compute ESP at a point due to a single multipole site
     * @param multipole The multipole object
     * @param site_position Position of the multipole site
     * @param eval_point Position where ESP is evaluated
     * @return ESP value in atomic units
     */
    double compute_esp_at_point(const occ::dma::Mult& multipole,
                               const Vec3& site_position,
                               const Vec3& eval_point) const;
    
    /**
     * Compute ESP at multiple points due to a single multipole site
     * @param multipole The multipole object  
     * @param site_position Position of the multipole site
     * @param eval_points Positions where ESP is evaluated
     * @return ESP values in atomic units
     */
    std::vector<double> compute_esp_at_points(const occ::dma::Mult& multipole,
                                            const Vec3& site_position,
                                            const std::vector<Vec3>& eval_points) const;
    
    /**
     * Compute interaction energy between two multipole sites
     * @param mult1 First multipole
     * @param pos1 Position of first multipole
     * @param mult2 Second multipole  
     * @param pos2 Position of second multipole
     * @return Interaction energy in atomic units
     */
    double compute_interaction_energy(const occ::dma::Mult& mult1,
                                    const Vec3& pos1,
                                    const occ::dma::Mult& mult2,
                                    const Vec3& pos2) const;

private:
    int m_max_rank;
    
    /**
     * Convert multipole moment to Orient's convention if needed
     * Orient uses different normalization/ordering than DMA 
     */
    std::vector<double> convert_multipole_to_orient(const occ::dma::Mult& mult) const;
    
    /**
     * Get the power of R for given multipole ranks following Orient's convention
     */
    int get_power_for_ranks(int l1, int l2) const;
};

} // namespace occ::mults