#include <occ/interaction/polarization_partitioning.h>
#include <occ/core/log.h>
#include <cmath>

namespace occ::interaction::polarization_partitioning {

namespace {

/**
 * @brief Computes gradient-based energy attribution for a single molecule
 * 
 * Following the reference Python implementation pattern:
 * - For total energy E = f(S) where S = sum(xi)
 * - Attribution for component xi: (E / |S|²) * (xi · S)
 * - For polarization: E_pol = -1/2 * sum_i α_i * |F_total_i|^2
 * - Attribution: (E_pol / sum_i(α_i * |F_total_i|²)) * sum_i(α_i * F_pair_i · F_total_i)
 * 
 * @param polarizabilities Atomic polarizabilities
 * @param total_field Total electric field at each atom
 * @param pair_field Field contribution from specific pair
 * @return Attributed energy contribution
 */
double compute_gradient_attribution(
    const Vec& polarizabilities,
    const Mat3N& total_field,
    const Mat3N& pair_field
) {
    const int num_atoms = polarizabilities.size();
    
    // First compute the total polarization energy and normalization factor
    double total_pol_energy = 0.0;
    double weighted_field_norm_sq = 0.0;
    
    for (int i = 0; i < num_atoms; ++i) {
        const Vec3 f_total = total_field.col(i);
        const double alpha_i = polarizabilities(i);
        const double field_sq = f_total.squaredNorm();
        
        total_pol_energy += -0.5 * alpha_i * field_sq;
        weighted_field_norm_sq += alpha_i * field_sq;
    }
    
    // Avoid division by zero
    if (std::abs(weighted_field_norm_sq) < 1e-12) {
        return 0.0;
    }
    
    // Compute the coefficient: c = E_total / (weighted norm squared)
    const double c = total_pol_energy / weighted_field_norm_sq;
    
    // Compute the attribution: c * sum_i(α_i * F_pair_i · F_total_i)
    double attributed_energy = 0.0;
    for (int i = 0; i < num_atoms; ++i) {
        const Vec3 f_total = total_field.col(i);
        const Vec3 f_pair = pair_field.col(i);
        const double alpha_i = polarizabilities(i);
        
        attributed_energy += alpha_i * f_pair.dot(f_total);
    }
    
    return c * attributed_energy;
}

} // anonymous namespace

PairContribution partition_crystal_polarization_energy(
    const core::Dimer& dimer,
    const Mat3N& total_field_a,
    const Mat3N& total_field_b, 
    const Mat3N& pair_field_a,
    const Mat3N& pair_field_b,
    const Vec& polarizabilities_a,
    const Vec& polarizabilities_b
) {
    PairContribution result;
    result.molecule_a_idx = dimer.a().asymmetric_molecule_idx();
    result.molecule_b_idx = dimer.b().asymmetric_molecule_idx();
    
    const int num_atoms_a = polarizabilities_a.size();
    const int num_atoms_b = polarizabilities_b.size();
    
    // Compute normalization coefficients for each molecule
    double total_pol_a = 0.0, weighted_norm_sq_a = 0.0;
    for (int i = 0; i < num_atoms_a; ++i) {
        const Vec3 f_total = total_field_a.col(i);
        const double field_sq = f_total.squaredNorm();
        total_pol_a += -0.5 * polarizabilities_a(i) * field_sq;
        weighted_norm_sq_a += polarizabilities_a(i) * field_sq;
    }
    const double c_a = (std::abs(weighted_norm_sq_a) > 1e-12) ? total_pol_a / weighted_norm_sq_a : 0.0;
    
    double total_pol_b = 0.0, weighted_norm_sq_b = 0.0;
    for (int i = 0; i < num_atoms_b; ++i) {
        const Vec3 f_total = total_field_b.col(i);
        const double field_sq = f_total.squaredNorm();
        total_pol_b += -0.5 * polarizabilities_b(i) * field_sq;
        weighted_norm_sq_b += polarizabilities_b(i) * field_sq;
    }
    const double c_b = (std::abs(weighted_norm_sq_b) > 1e-12) ? total_pol_b / weighted_norm_sq_b : 0.0;
    
    // Store detailed atom-level contributions for molecule A
    result.atoms_a.reserve(num_atoms_a);
    for (int i = 0; i < num_atoms_a; ++i) {
        AtomContribution contrib;
        contrib.atom_idx = i;
        contrib.polarizability = polarizabilities_a(i);
        contrib.total_field = total_field_a.col(i);
        contrib.pair_field = pair_field_a.col(i);
        
        // Gradient-based attribution: c_a * α_i * (F_pair_i · F_total_i)
        contrib.energy = c_a * contrib.polarizability * contrib.pair_field.dot(contrib.total_field);
        
        result.atoms_a.push_back(contrib);
    }
    
    // Store detailed atom-level contributions for molecule B
    result.atoms_b.reserve(num_atoms_b);
    for (int i = 0; i < num_atoms_b; ++i) {
        AtomContribution contrib;
        contrib.atom_idx = i;
        contrib.polarizability = polarizabilities_b(i);
        contrib.total_field = total_field_b.col(i);
        contrib.pair_field = pair_field_b.col(i);
        
        // Gradient-based attribution: c_b * α_i * (F_pair_i · F_total_i)
        contrib.energy = c_b * contrib.polarizability * contrib.pair_field.dot(contrib.total_field);
        
        result.atoms_b.push_back(contrib);
    }
    
    // Calculate total pair energy using the helper function
    result.total_energy = compute_gradient_attribution(polarizabilities_a, total_field_a, pair_field_a) +
                         compute_gradient_attribution(polarizabilities_b, total_field_b, pair_field_b);
    
    occ::log::debug("Partitioned polarization energy for dimer {}-{}: {:.6f} au",
                    result.molecule_a_idx, result.molecule_b_idx, result.total_energy);
    
    return result;
}

std::vector<PairContribution> partition_all_pairs(
    const std::vector<core::Dimer>& dimers,
    const FieldMap& total_fields,
    const FieldMap& pair_fields,
    const std::vector<Vec>& polarizabilities
) {
    std::vector<PairContribution> results;
    results.reserve(dimers.size());
    
    for (const auto& dimer : dimers) {
        size_t a_idx = dimer.a().asymmetric_molecule_idx();
        size_t b_idx = dimer.b().asymmetric_molecule_idx();
        
        auto pair_key = std::make_pair(a_idx, b_idx);
        
        if (total_fields.find(pair_key) == total_fields.end() ||
            pair_fields.find(pair_key) == pair_fields.end()) {
            occ::log::warn("Missing field data for dimer {}-{}", a_idx, b_idx);
            continue;
        }
        
        const Mat3N& total_field_a = total_fields.at(pair_key);
        const Mat3N& total_field_b = total_fields.at(std::make_pair(b_idx, a_idx));
        const Mat3N& pair_field_a = pair_fields.at(pair_key);
        const Mat3N& pair_field_b = pair_fields.at(std::make_pair(b_idx, a_idx));
        
        auto contribution = partition_crystal_polarization_energy(
            dimer, total_field_a, total_field_b, 
            pair_field_a, pair_field_b,
            polarizabilities[a_idx], polarizabilities[b_idx]
        );
        
        results.push_back(contribution);
    }
    
    return results;
}

std::vector<CouplingTerm> compute_coupling_terms(
    const std::vector<Mat3N>& neighbor_fields,
    const std::vector<size_t>& neighbor_indices,
    const Vec& polarizabilities
) {
    std::vector<CouplingTerm> couplings;
    const size_t num_neighbors = neighbor_fields.size();

    if (num_neighbors < 2) {
        return couplings; // No coupling terms if less than 2 neighbors
    }

    // Reserve space for all pairs: N*(N-1)/2
    couplings.reserve(num_neighbors * (num_neighbors - 1) / 2);

    // Compute coupling for each pair of neighbors
    for (size_t b = 0; b < num_neighbors; ++b) {
        for (size_t c = b + 1; c < num_neighbors; ++c) {
            const Mat3N& field_b = neighbor_fields[b];
            const Mat3N& field_c = neighbor_fields[c];
            const int num_atoms = polarizabilities.size();

            // Compute C_BC = -Σ_i α_i (F_B,i · F_C,i)
            double coupling = 0.0;
            for (int i = 0; i < num_atoms; ++i) {
                const Vec3 f_b = field_b.col(i);
                const Vec3 f_c = field_c.col(i);
                coupling += polarizabilities(i) * f_b.dot(f_c);
            }
            coupling = -coupling; // Apply negative sign

            CouplingTerm term;
            term.neighbor_b_idx = neighbor_indices[b];
            term.neighbor_c_idx = neighbor_indices[c];
            term.coupling_energy = coupling;

            couplings.push_back(term);

            occ::log::debug("Coupling term between neighbors {} and {}: {:.6f} au",
                           neighbor_indices[b], neighbor_indices[c], coupling);
        }
    }

    occ::log::debug("Computed {} coupling terms for {} neighbors",
                   couplings.size(), num_neighbors);

    return couplings;
}

} // namespace occ::interaction::polarization_partitioning