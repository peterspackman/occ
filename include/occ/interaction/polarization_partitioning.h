#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/dimer.h>
#include <ankerl/unordered_dense.h>
#include <vector>

namespace occ::interaction::polarization_partitioning {

/**
 * @brief Represents the contribution of a single atom to polarization energy attribution
 */
struct AtomContribution {
    size_t atom_idx;              ///< Index of the atom within its molecule
    double energy;                ///< Attributed polarization energy for this atom
    Vec3 total_field;            ///< Total electric field at this atom
    Vec3 pair_field;             ///< Field contribution from the specific pair
    double polarizability;       ///< Atomic polarizability
};

/**
 * @brief Represents the partitioned polarization energy contribution from a dimer pair
 */
struct PairContribution {
    size_t molecule_a_idx;                      ///< Index of molecule A
    size_t molecule_b_idx;                      ///< Index of molecule B
    std::vector<AtomContribution> atoms_a;      ///< Atom-level contributions in molecule A
    std::vector<AtomContribution> atoms_b;      ///< Atom-level contributions in molecule B
    double total_energy;                        ///< Total attributed pair energy (atomic units)
};

/**
 * @brief Represents a pairwise coupling term between two neighbor interactions
 *
 * For central molecule A with neighbors B and C:
 * coupling_energy = -Σ_i α_i (F_B,i · F_C,i)
 *
 * This captures field reinforcement (positive) or cancellation (negative)
 */
struct CouplingTerm {
    size_t neighbor_b_idx;     ///< Index of neighbor B (unique dimer index)
    size_t neighbor_c_idx;     ///< Index of neighbor C (unique dimer index)
    double coupling_energy;    ///< Coupling energy in atomic units
};

/**
 * @brief Results from coupling term calculation for a central molecule
 */
struct MoleculeCouplingResults {
    size_t molecule_idx;                     ///< Index of central molecule
    std::vector<CouplingTerm> couplings;     ///< All pairwise coupling terms for this molecule
};

using FieldMap = ankerl::unordered_dense::map<std::pair<size_t, size_t>, Mat3N>;

/**
 * @brief Partitions crystal field polarization energy back to individual dimer pairs
 * 
 * Uses gradient-based energy attribution following the approach from the reference
 * Python implementation. The total crystal field polarization energy is smaller
 * than the sum of individual pair polarizations due to field cancellation effects.
 * 
 * @param dimer The molecular dimer pair
 * @param total_field_a Total crystal field on molecule A atoms  
 * @param total_field_b Total crystal field on molecule B atoms
 * @param pair_field_a Field contribution from molecule B onto A
 * @param pair_field_b Field contribution from molecule A onto B
 * @param polarizabilities_a Atomic polarizabilities for molecule A
 * @param polarizabilities_b Atomic polarizabilities for molecule B
 * @return PairContribution containing attributed energies
 */
PairContribution partition_crystal_polarization_energy(
    const core::Dimer& dimer,
    const Mat3N& total_field_a,
    const Mat3N& total_field_b, 
    const Mat3N& pair_field_a,
    const Mat3N& pair_field_b,
    const Vec& polarizabilities_a,
    const Vec& polarizabilities_b
);

/**
 * @brief Partition multiple dimer pairs simultaneously
 */
std::vector<PairContribution> partition_all_pairs(
    const std::vector<core::Dimer>& dimers,
    const FieldMap& total_fields,
    const FieldMap& pair_fields,
    const std::vector<Vec>& polarizabilities
);

/**
 * @brief Compute coupling terms between all pairs of neighbors for a molecule
 *
 * For each pair of neighbors (B,C) of a central molecule, computes:
 * C_BC = -Σ_i α_i (F_B,i · F_C,i)
 *
 * where i runs over atoms in the central molecule, F_B,i is the field at atom i
 * from neighbor B, and α_i is the atomic polarizability.
 *
 * @param neighbor_fields Vector of field contributions from each neighbor (Mat3N)
 * @param neighbor_indices Vector of unique dimer indices for each neighbor
 * @param polarizabilities Atomic polarizabilities for the central molecule
 * @return Vector of coupling terms for all neighbor pairs
 */
std::vector<CouplingTerm> compute_coupling_terms(
    const std::vector<Mat3N>& neighbor_fields,
    const std::vector<size_t>& neighbor_indices,
    const Vec& polarizabilities
);

} // namespace occ::interaction::polarization_partitioning