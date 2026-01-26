#include <occ/mults/esp.h>
#include <occ/mults/rotation.h>
#include <stdexcept>
#include <cmath>
#include <fmt/core.h>

namespace occ::mults {

MultipoleESP::MultipoleESP(int max_rank) : m_max_rank(max_rank) {
    if (max_rank < 0 || max_rank > 5) {
        throw std::invalid_argument("max_rank must be between 0 and 5");
    }
}

double MultipoleESP::compute_esp_at_point(const occ::dma::Mult& multipole,
                                         const Vec3& site_position,
                                         const Vec3& eval_point) const {
    
    // ESP calculation: potential at eval_point due to multipole at site_position
    // This is equivalent to the interaction energy between a unit charge at eval_point
    // and the multipole at site_position
    //
    // CRITICAL: Orient's convention for ESP is ra=multipole_site, rb=field_point
    // This makes the inter-site vector point FROM multipole TO field point
    // giving rax = er[0] where er = (field_point - multipole) / |distance|

    SFunctions sf(m_max_rank);
    sf.set_coordinates(site_position, eval_point); // A=multipole, B=field_point (Orient convention)
    
    double esp = 0.0;
    
    // Iterate over all multipole components 
    for (int t2 = 0; t2 < multipole.num_components(); t2++) {
        double q2 = multipole.q(t2);
        if (std::abs(q2) < 1e-15) continue; // Skip negligible components
        
        // Get l,m indices for this multipole component
        auto [l2, m2] = sf.index_to_lm(t2);

        // For ESP, we compute interaction between multipole and unit charge at field point
        // With Orient convention: Site A = multipole, Site B = field point (charge)
        // For multipole-charge interactions, j should equal the multipole rank l2
        auto result = sf.compute_s_function(t2, 0, l2, 0); // t1=t2 (multipole@A), t2=0 (charge@B), j=l2
        
        // Removed scattered debug - using consolidated Orient-style debug below
        
        // Power of R for interaction is (l1 + l2 + 1) = (0 + l2 + 1) = l2 + 1
        int power = l2 + 1;
        double r = sf.r();
        
        if (r < 1e-15) {
            throw std::runtime_error("Division by zero: evaluation point too close to multipole site");
        }
        
        double r_power = std::pow(r, power);
        
        // ESP contribution: (unit charge) * q2 * S-function / r^power
        double contribution = q2 * result.s0 / r_power;
        esp += contribution;
    }

    return esp;
}

std::vector<double> MultipoleESP::compute_esp_at_points(const occ::dma::Mult& multipole,
                                                       const Vec3& site_position,
                                                       const std::vector<Vec3>& eval_points) const {
    std::vector<double> results;
    results.reserve(eval_points.size());
    
    for (const auto& point : eval_points) {
        results.push_back(compute_esp_at_point(multipole, site_position, point));
    }
    
    return results;
}

double MultipoleESP::compute_interaction_energy(const occ::dma::Mult& mult1,
                                              const Vec3& pos1,
                                              const occ::dma::Mult& mult2,
                                              const Vec3& pos2) const {

    // Compute interaction energy without rotation - Orient's approach
    // S-functions handle arbitrary geometry directly using rax, ray, raz, rbx, rby, rbz

    // Compute inter-site vector and distance
    Vec3 rab = pos2 - pos1;
    double r = rab.norm();

    if (r < 1e-15) {
        throw std::runtime_error("Sites too close for interaction energy calculation");
    }

    // Set up S-functions with actual positions (no rotation)
    SFunctions sf(m_max_rank);
    sf.set_coordinates(pos1, pos2);

    // Binomial coefficients for fac(n) = binom(j, l1) factor
    occ::dma::BinomialCoefficients binomial(m_max_rank + 4);

    double energy = 0.0;

    // Triple loop: over multipole components at site A, at site B
    // Following Orient's convention
    for (int t1 = 0; t1 < mult1.num_components(); t1++) {
        double q1 = mult1.q(t1);
        if (std::abs(q1) < 1e-15) continue;

        auto [l1, m1] = sf.index_to_lm(t1);

        for (int t2 = 0; t2 < mult2.num_components(); t2++) {
            double q2 = mult2.q(t2);
            if (std::abs(q2) < 1e-15) continue;

            auto [l2, m2] = sf.index_to_lm(t2);

            // Orient uses j = l1 + l2 for electrostatic interactions (jdiff = 0)
            // This corresponds to the leading-order 1/R^(l1+l2+1) term
            int j = l1 + l2;

            // Orient limits electrostatic interactions to R^(-eslimit)
            // eslimit = max_rank + 1, so j <= max_rank (since power = j+1)
            // Skip terms with j > max_rank to respect the JSON max_rank setting
            if (j > m_max_rank) continue;

            // Calculate S-function for this (t1, t2, j) combination
            auto result = sf.compute_s_function(t1, t2, j, 0);

            // Power of R is (l1 + l2 + 1)
            int power = l1 + l2 + 1;
            double r = sf.r();
            double r_power = std::pow(r, power);

            // Orient's coupling coefficient: fac(n) = binom(j, l1) = binom(l1+l2, l1)
            // This accounts for the degeneracy of multipole combinations
            double fac = binomial.binomial(j, l1);

            // Add contribution: fac * q1 * q2 * S-function / r^power
            double contrib = fac * q1 * q2 * result.s0 / r_power;

            // DEBUG: Print details for quadrupole-quadrupole
            if (l1 == 2 && l2 == 2) {
                fmt::print("OCC: t1={:2} t2={:2} j={} fac={:.1f} qq={:12.8f} s0={:12.8f} contrib={:14.10f}\n",
                    t1, t2, j, fac, q1*q2, result.s0, contrib);
            }

            // DEBUG: Print details for octapole-dipole and dipole-octapole
            if ((l1 == 3 && l2 == 1) || (l1 == 1 && l2 == 3)) {
                fmt::print("OCC: t1={:2} t2={:2} j={} l1={} l2={} fac={:.1f} qq={:12.8f} s0={:12.8f} contrib={:14.10f}\n",
                    t1, t2, j, l1, l2, fac, q1*q2, result.s0, contrib);
            }

            // DEBUG: Print details for octapole-quadrupole
            if ((l1 == 3 && l2 == 2) || (l1 == 2 && l2 == 3)) {
                fmt::print("OCC: t1={:2} t2={:2} j={} l1={} l2={} fac={:.1f} qq={:12.8f} s0={:12.8f} contrib={:14.10f}\n",
                    t1, t2, j, l1, l2, fac, q1*q2, result.s0, contrib);
            }

            // DEBUG: Print details for hexadecapole-dipole and dipole-hexadecapole
            if ((l1 == 4 && l2 == 1) || (l1 == 1 && l2 == 4)) {
                fmt::print("OCC: t1={:2} t2={:2} j={} l1={} l2={} fac={:.1f} qq={:12.8f} s0={:12.8f} contrib={:14.10f}\n",
                    t1, t2, j, l1, l2, fac, q1*q2, result.s0, contrib);
            }

            energy += contrib;
        }
    }

    return energy;
}

std::vector<double> MultipoleESP::convert_multipole_to_orient(const occ::dma::Mult& mult) const {
    // For now, assume our DMA multipole convention matches Orient's
    // This may need adjustment based on detailed comparison
    std::vector<double> orient_q(mult.num_components());
    for (int i = 0; i < mult.num_components(); i++) {
        orient_q[i] = mult.q(i);
    }
    return orient_q;
}

int MultipoleESP::get_power_for_ranks(int l1, int l2) const {
    // Following Orient: power = l1 + l2 + 1
    return l1 + l2 + 1;
}

} // namespace occ::mults