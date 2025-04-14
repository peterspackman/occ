#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/add_qlm.h>
#include <vector>

namespace occ::dma {

/**
 * @brief Move multipoles to the nearest site (but if two or more sites are
 *        almost equidistant, move a fraction to each).
 *
 * @param x X-coordinate of the point to move from
 * @param y Y-coordinate of the point to move from
 * @param z Z-coordinate of the point to move from
 * @param qt Temporary multipole moments to be moved
 * @param site_positions Matrix of site positions (3 x n_sites)
 * @param site_radii Vector of site radii
 * @param site_limits Vector of maximum multipole ranks for each site
 * @param q Array of multipole moments for each site
 * @param lmax Maximum multipole rank
 */
void moveq(Eigen::Ref<const Vec3> pos, Mult &qt, const Mat3N &site_positions,
           const Vec &site_radii, const IVec &site_limits, std::vector<Mult> &q,
           int lmax);

} // namespace occ::dma
