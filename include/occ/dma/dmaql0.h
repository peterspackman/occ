#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/qm/wavefunction.h>
#include <vector>

namespace occ::dma {

/**
 * @brief Calculate multipole moments for linear molecules and shift them to the
 * nearest site.
 *
 * This is a direct implementation of the dmaql0 subroutine from the GDMA
 * Fortran code. It calculates multipole moments for linear molecules, where
 * only the Qlm with m=0 are non-zero and stored in the order Q0, Q1, Q2, ...
 *
 * @param wfn Wavefunction containing basis set and density matrix
 * @param max_rank Maximum multipole rank to calculate
 * @param include_nuclei Whether to include nuclear contributions in the
 * calculation
 * @param use_slices Whether to use slices of space for calculating multipoles
 * @return std::vector<Mult> Multipole moments for each site
 */
std::vector<Mult> dmaql0(const occ::qm::Wavefunction &wfn, int max_rank = 4,
                         bool include_nuclei = true,
                         bool use_slices = false);

} // namespace occ::dma
