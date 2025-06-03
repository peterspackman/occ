#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/binomial.h>
#include <occ/dma/solid_harmonics.h>
#include <occ/dma/dma.h> // Ensure DMASites and Mult are defined
#include <occ/core/linear_algebra.h> // Ensure Vec3, Mat, etc. are defined
#include <occ/core/linear_algebra.h> // Ensure Vec3, Mat, etc. are defined
#include <vector>
#include <Eigen/Dense>

namespace occ::dma {


class MultipoleShifter {
public:
  MultipoleShifter(const Vec3 &pos, Mult &qt,
                   const DMASites &sites, std::vector<Mult> &q, int lmax);
  void shift();

  void shift_multipoles(const Mult &q1, int l1, int m1, Mult &q2, int m2,
                        const Vec3 &pos);

private:
  int find_nearest_site_with_limit(int low, int start) const;
  bool direct_transfer(int k, int t1, int t2);
  bool distributed_transfer(int k, int low, int t1, int t2, int lp1sq,
                            double eps);
  bool process_site(int k, int low, int t1, int t2, int lp1sq, double eps);

  // Multipole shifting functionality (migrated from shiftq.cpp)
  int estimate_largest_transferred_multipole(const Vec3 &pos,
                                             const Mult &mult, int l, int m1,
                                             int m2, double eps);
  Mat get_cplx_sh(const Vec3 &pos, int N);
  Mat get_cplx_mults(const Mult &mult, int l1, int m1, int N);

  Vec3 m_pos;
  DMASites m_sites;
  Mult &m_qt;
  std::vector<Mult> &m_q;
  int m_lmax;
  int m_num_sites{0};
  Vec m_rr;
  std::vector<int> m_destination_sites;
  int m_site_with_highest_limit{0};
};

} // namespace occ::dma
