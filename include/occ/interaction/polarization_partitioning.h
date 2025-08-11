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

} // namespace occ::interaction::polarization_partitioning