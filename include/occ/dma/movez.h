#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <vector>

namespace occ::dma {

/**
 * @brief Shift multipoles from one center to another along z-axis
 *
 * This is equivalent to the SHIFTZ subroutine in the GDMA Fortran code.
 * It shifts multipole moments from one center to another along the z-axis.
 *
 * @param q1 Source multipoles
 * @param l1 Minimum rank to shift
 * @param m1 Maximum rank to shift from source
 * @param q2 Destination multipoles
 * @param m2 Maximum rank to keep at destination
 * @param z Displacement along z-axis
 */
void shiftz(const Mult &q1, int l1, int m1, Mult &q2, int m2, double z);
/**
 * @brief Move multipoles from source position to nearest site
 *
 * This is equivalent to the MOVEZ subroutine in the GDMA Fortran code.
 * It moves the multipole contributions to the nearest expansion site.
 *
 * @param qp Multipole moments to be moved
 * @param p Source position
 * @param sites Matrix of site positions
 * @param site_radii Vector of site radii
 * @param site_limits Vector of maximum rank for each site
 * @param site_multipoles Vector of multipoles at each site
 * @param max_rank Maximum multipole rank
 */
void movez(Mult &qp, double p, const Mat3N &sites,
           const Vec &site_radii, const IVec &site_limits,
           std::vector<Mult> &site_multipoles, int max_rank);
} // namespace occ::dma
