#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/dma/multipole.h>
#include <occ/qm/shell.h>
#include <occ/qm/wavefunction.h>
#include <vector>

namespace occ::dma {

using occ::core::Atom;
using occ::qm::AOBasis;
using occ::qm::Shell;
using occ::qm::Wavefunction;

/**
 * @brief Distributed Multipole Analysis (DMA) result for a molecule
 * 
 * Contains the multipole moments distributed over atomic sites
 */
struct DMAResult {
    std::vector<Multipole> multipoles; // Multipoles for each site
    Mat3N positions;                   // Positions of each site
    IVec atom_indices;                 // Index of atom each site belongs to
    Vec radii;                         // Radii used for each site
    int max_rank{0};                   // Maximum rank of multipoles
    
    // Get total multipole of specific rank
    Multipole total_multipole() const;
    
    // Get total dipole moment (in atomic units)
    Vec3 total_dipole() const;
    
    // Get total quadrupole moment (in atomic units)
    Mat3 total_quadrupole() const;
};

/**
 * @brief Perform Distributed Multipole Analysis (DMA) on a wavefunction
 * 
 * This implements the DMA method as described by Stone (2005)
 * 
 * @param wfn The wavefunction to analyze
 * @param max_rank Maximum rank of multipoles to compute (0=charges, 1=dipoles, 2=quadrupoles, etc.)
 * @param use_grid Whether to use numerical quadrature on grid points (DMA4)
 * @param grid_level Grid level for numerical integration (1-5)
 * @return DMAResult containing distributed multipoles
 */
DMAResult distributed_multipole_analysis(
    const Wavefunction &wfn,
    int max_rank = 2,
    bool use_grid = true,
    int grid_level = 4
);

/**
 * @brief Decontracts a contracted shell into primitive shells
 * 
 * @param shell The contracted shell to decontract
 * @return std::vector<Shell> Vector of primitive shells
 */
std::vector<Shell> decontract_shell(const Shell &shell);

/**
 * @brief Decontracts an entire basis set into primitive shells
 * 
 * @param basis The contracted basis to decontract
 * @return AOBasis Decontracted basis with primitive shells
 */
AOBasis decontract_basis(const AOBasis &basis);

} // namespace occ::dma
