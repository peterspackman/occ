#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/binomial.h>
#include <occ/dma/solid_harmonics.h>
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
  
  // Multipole shifting functionality (migrated from shiftq.cpp)
  int estimate_largest_transferred_multipole(Eigen::Ref<const Vec3> pos,
                                           const Mult &mult, int l, int m1, int m2,
                                           double eps);
  Mat get_cplx_sh(Eigen::Ref<const Vec3> pos, int N);
  Mat get_cplx_mults(const Mult &mult, int l1, int m1, int N);
  void shift_multipoles(const Mult &q1, int l1, int m1, Mult &q2, int m2,
                        Eigen::Ref<const Vec3> pos);

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


} // namespace occ::dma
