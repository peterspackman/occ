#include <cmath>
#include <occ/dma/add_qlm.h>
#include <occ/dma/moveq.h>
#include <occ/dma/shiftq.h>

namespace occ::dma {

MultipoleShifter::MultipoleShifter(Eigen::Ref<const Vec3> pos, Mult &qt,
                                   const Mat3N &site_positions,
                                   const Vec &site_radii,
                                   const IVec &site_limits,
                                   std::vector<Mult> &q, int lmax)
    : m_pos(pos), m_site_radii(site_radii), m_site_limits(site_limits),
      m_site_positions(site_positions), m_qt(qt), m_q(q), m_lmax(lmax),
      m_num_sites(site_positions.cols()), m_rr(m_num_sites) {
  m_destination_sites.reserve(m_num_sites);

  for (int i = 0; i < m_num_sites; i++) {
    m_rr(i) = (m_pos - m_site_positions.col(i)).squaredNorm() /
            (m_site_radii(i) * m_site_radii(i));

    if (site_limits(i) > site_limits(m_site_with_highest_limit)) {
      m_site_with_highest_limit = i;
    }
  }
}

void MultipoleShifter::shift() {
  constexpr double eps = 1e-6;
  int lp1sq = (m_lmax + 1) * (m_lmax + 1);
  int low = 0;

  while (true) {
    int k = find_nearest_site_with_limit(low, m_site_with_highest_limit);
    int t1 = low * low;
    int t2 = (m_site_limits(k) + 1) * (m_site_limits(k) + 1);

    bool completed = process_site(k, low, t1, t2, lp1sq, eps);
    if (completed) {
      return;
    }

    low = m_site_limits(k) + 1;
  }
}


int MultipoleShifter::find_nearest_site_with_limit(int low, int start) const {
  int k = start;
  for (int i = 0; i < m_rr.rows(); i++) {
    if (m_rr(i) < m_rr(k) && m_site_limits(i) >= low) {
      k = i;
    }
  }
  return k;
}

bool MultipoleShifter::direct_transfer(int k, int t1, int t2) {
  for (int i = t1; i < t2; i++) {
    m_q[k].q(i) += m_qt.q(i);
    m_qt.q(i) = 0.0;
  }

  return m_site_limits(k) >= m_lmax;
}

bool MultipoleShifter::distributed_transfer(int k, int low, int t1, int t2, int lp1sq, double eps) {
  // Find all sites at approximately the same distance
  m_destination_sites.clear();
  m_destination_sites.push_back(k);
  for (int i = 0; i < m_num_sites; i++) {
    if (i == k || m_rr(i) > m_rr(k) + eps || m_site_limits(i) != m_site_limits(k) ||
        m_site_limits(i) < low) {
      continue;
    }
    m_destination_sites.push_back(i);
  }

  // If multiple equidistant sites, distribute equally
  if (m_destination_sites.size() > 1) {
    m_qt.q.array() *= 1.0 / m_destination_sites.size();
  }

  // Shift multipoles to each site
  for (int site_idx : m_destination_sites) {
    // Call shiftq to transfer multipoles
    shiftq(m_qt, low, m_site_limits(site_idx), m_q[site_idx], m_lmax,
           m_pos - m_site_positions.col(site_idx));
  }

  // Zero out the transferred multipoles
  for (int i = t1; i < t2; i++) {
    m_qt.q(i) = 0.0;
  }

  // If we've reached lmax at this site, we're done
  if (m_site_limits(k) >= m_lmax) {
    return true;
  }

  // Transfer higher-rank multipoles back to qt
  for (int site_idx : m_destination_sites) {
    // Call shiftq to transfer higher-rank multipoles back to qt
    shiftq(m_q[site_idx], m_site_limits(site_idx) + 1, m_lmax, m_qt, m_lmax,
           m_site_positions.col(site_idx) - m_pos);
    // Zero out transferred multipoles at the site
    m_q[site_idx].q.segment(t2, lp1sq - t2).setZero();
  }

  return false;
}

bool MultipoleShifter::process_site(int k, int low, int t1, int t2, int lp1sq, double eps) {
  // If very close to a site, add all multipoles directly
  if (m_rr(k) <= eps) {
    return direct_transfer(k, t1, t2);
  } else {
    return distributed_transfer(k, low, t1, t2, lp1sq, eps);
  }
}


} // namespace occ::dma
