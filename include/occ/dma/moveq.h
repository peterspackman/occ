#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/add_qlm.h>
#include <vector>

namespace occ::dma {

class MultipoleShifter {
public:
  MultipoleShifter(Eigen::Ref<const Vec3> pos, Mult &qt,
                   const Mat3N &site_positions, const Vec &site_radii,
                   const IVec &site_limits, std::vector<Mult> &q, int lmax);
  void shift();

private:
  int find_nearest_site_with_limit(int low, int start) const;
  bool direct_transfer(int k, int t1, int t2);
  bool distributed_transfer(int k, int low, int t1, int t2, int lp1sq, double eps);
  bool process_site(int k, int low, int t1, int t2, int lp1sq, double eps);

  Eigen::Ref<const Vec3> m_pos;
  Eigen::Ref<const Vec> m_site_radii;
  Eigen::Ref<const IVec> m_site_limits;
  Eigen::Ref<const Mat3N> m_site_positions;
  Mult &m_qt;
  std::vector<Mult> &m_q;
  int m_lmax;
  int m_num_sites{0};
  Vec m_rr;
  std::vector<int> m_destination_sites;
  int m_site_with_highest_limit{0};
};

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
