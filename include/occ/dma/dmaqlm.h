#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/qm/wavefunction.h>
#include <vector>

namespace occ::dma {

/**
 * @brief Calculate multipole moments and shift them to the nearest site.
 *
 * This is a direct implementation of the dmaqlm subroutine from the GDMA
 * Fortran code. It calculates multipole moments for all pairs of basis
 * functions and shifts them to appropriate expansion sites.
 *
 * @param wfn Wavefunction containing basis set and density matrix
 * @param max_rank Maximum multipole rank to calculate
 * @param verbose Controls output verbosity
 * @param include_nuclei Whether to include nuclear contributions in the
 * calculation
 * @param bigexp Threshold for switching between analytical and grid-based
 * methods
 * @return std::vector<Mult> Multipole moments for each site
 */
std::vector<Mult> dmaqlm(const occ::qm::Wavefunction &wfn, int max_rank = 4,
                         bool include_nuclei = true, double bigexp = 4.0);

} // namespace occ::dma
