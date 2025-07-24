#include <cmath>
#include <limits>
#include <occ/core/log.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/multipole_shifter.h>

namespace occ::dma {

MultipoleShifter::MultipoleShifter(const Vec3 &pos, Mult &qt,
                                   const DMASites &sites, std::vector<Mult> &q,
                                   int lmax)
    : m_pos(pos), m_sites(sites), m_qt(qt), m_q(q), m_lmax(lmax),
      m_num_sites(m_sites.positions.cols()), m_rr(m_num_sites) {
  m_destination_sites.reserve(m_num_sites);

  for (int i = 0; i < m_num_sites; i++) {
    m_rr(i) = (m_pos - m_sites.positions.col(i)).squaredNorm() /
              (m_sites.radii(i) * m_sites.radii(i));

    if (m_sites.limits(i) > m_sites.limits(m_site_with_highest_limit)) {
      m_site_with_highest_limit = i;
    }
  }
}

void MultipoleShifter::shift() {
  constexpr double eps = 1e-6;
  int lp1sq = (m_lmax + 1) * (m_lmax + 1);
  int low = 0;
  int iteration = 0;

  while (true) {
    iteration++;
    if (iteration > 100) {
      log::error("MultipoleShifter.shift() infinite loop detected after {} iterations", iteration);
      log::error("low={}, m_site_with_highest_limit={}", low, m_site_with_highest_limit);
      for (int i = 0; i < m_sites.size(); i++) {
        log::error("Site {}: limit={}", i, m_sites.limits(i));
      }
      break;
    }
    
    int k = find_nearest_site_with_limit(low, m_site_with_highest_limit);
    
    // Check if no valid site was found
    if (k < 0) {
      break;
    }
    
    int t1 = low * low;
    int t2 = (m_sites.limits(k) + 1) * (m_sites.limits(k) + 1);

    bool completed = process_site(k, low, t1, t2, lp1sq, eps);
    if (completed) {
      return;
    }

    low = m_sites.limits(k) + 1;
  }
}

int MultipoleShifter::find_nearest_site_with_limit(int low, int start) const {
  int k = -1;  // Initialize to invalid index
  double min_rr = std::numeric_limits<double>::max();
  
  // Find the nearest site that meets the limit criteria
  for (int i = 0; i < m_rr.rows(); i++) {
    if (m_sites.limits(i) >= low && m_rr(i) < min_rr) {
      min_rr = m_rr(i);
      k = i;
    }
  }
  
  return k;  // Returns -1 if no valid site found
}

bool MultipoleShifter::direct_transfer(int k, int t1, int t2) {
  for (int i = t1; i < t2; i++) {
    m_q[k].q(i) += m_qt.q(i);
    m_qt.q(i) = 0.0;
  }

  return m_sites.limits(k) >= m_lmax;
}

bool MultipoleShifter::distributed_transfer(int k, int low, int t1, int t2,
                                            int lp1sq, double eps) {
  // Find all sites at approximately the same distance
  m_destination_sites.clear();
  m_destination_sites.push_back(k);
  for (int i = 0; i < m_num_sites; i++) {
    if (i == k || m_rr(i) > m_rr(k) + eps ||
        m_sites.limits(i) != m_sites.limits(k) || m_sites.limits(i) < low) {
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
    // Call shift_multipoles to transfer multipoles
    shift_multipoles(m_qt, low, m_sites.limits(site_idx), m_q[site_idx], m_lmax,
                     m_pos - m_sites.positions.col(site_idx));
  }

  // Zero out the transferred multipoles
  for (int i = t1; i < t2; i++) {
    m_qt.q(i) = 0.0;
  }

  // If we've reached lmax at this site, we're done
  if (m_sites.limits(k) >= m_lmax) {
    return true;
  }

  // Transfer higher-rank multipoles back to qt
  for (int site_idx : m_destination_sites) {
    // Call shift_multipoles to transfer higher-rank multipoles back to qt
    shift_multipoles(m_q[site_idx], m_sites.limits(site_idx) + 1, m_lmax, m_qt,
                     m_lmax, m_sites.positions.col(site_idx) - m_pos);
    // Zero out transferred multipoles at the site
    m_q[site_idx].q.segment(t2, lp1sq - t2).setZero();
  }

  return false;
}

bool MultipoleShifter::process_site(int k, int low, int t1, int t2, int lp1sq,
                                    double eps) {
  // If very close to a site, add all multipoles directly
  if (m_rr(k) <= eps) {
    return direct_transfer(k, t1, t2);
  } else {
    return distributed_transfer(k, low, t1, t2, lp1sq, eps);
  }
}

// Migrated multipole shifting functionality (from shiftq.cpp)

int MultipoleShifter::estimate_largest_transferred_multipole(
    const Vec3 &pos, const Mult &mult, int l, int m1, int m2, double eps) {
  if (eps == 0.0 || m2 < 1)
    return m2;
  double r2 = 4.0 * pos.squaredNorm();
  int n = 0;
  double a = 0.0;

  if (l == 0)
    a = mult.Q00() * mult.Q00();

  for (int k = std::max(1, l); k <= m2; k++) {
    a *= r2;
    if (k <= m1) {
      int t1 = k * k + 1; // Use 1-based indexing for temporary calculations
      int t2 = (k + 1) * (k + 1);
      for (int i = t1; i <= t2; i++) {
        a += mult.q(i - 1) * mult.q(i - 1); // Adjust index when accessing q1
      }
    }
    if (a > eps)
      n = k;
  }
  return n;
}

Mat MultipoleShifter::get_cplx_sh(const Vec3 &pos, int N) {
  constexpr double RTHALF = 0.7071067811865475244;

  size_t num_components = (N + 1) * (N + 1) + 1;
  // Evaluate solid harmonics in real form
  Vec r = Vec::Zero(num_components); // One extra element
  solid_harmonics(pos, N, r);

  // Construct complex solid harmonics RC
  // RC[i][0] = real part, RC[i][1] = imaginary part
  Mat rc = Mat::Zero(num_components, 2); // One extra element/row
  rc(1, 0) = r(1);                       // Now we can use 1-based indexing
  rc(1, 1) = 0.0;

  for (int k = 1; k <= N; k++) {
    int kb = k * k + k + 1; // Use the same formula as Fortran
    int km = k * k + 1;

    rc(kb, 0) = r(km);
    rc(kb, 1) = 0.0;
    km++;

    double s = RTHALF;
    for (int m = 1; m <= k; m++) {
      s = -s;
      rc(kb - m, 0) = RTHALF * r(km);
      rc(kb - m, 1) = -RTHALF * r(km + 1);
      rc(kb + m, 0) = s * r(km);
      rc(kb + m, 1) = s * r(km + 1);
      km += 2;
    }
  }
  return rc;
}

Mat MultipoleShifter::get_cplx_mults(const Mult &mult, int l1, int m1, int N) {
  constexpr double RTHALF = 0.7071067811865475244;
  size_t num_components = (N + 1) * (N + 1) + 1;
  int k1 = std::max(1, l1);

  Mat qc = Mat::Zero(num_components, 2); // One extra element/row
  if (l1 == 0) {
    qc(1, 0) = mult.Q00(); // Use index 1 for first element
    qc(1, 1) = 0.0;
  }

  if (m1 > 0) {
    for (int k = k1; k <= m1; k++) {
      int kb = k * k + k + 1;
      int km = k * k + 1;

      qc(kb, 0) = mult.q(km - 1); // Adjust index when accessing mult
      qc(kb, 1) = 0.0;
      km++;

      double s = RTHALF;
      for (int m = 1; m <= k; m++) {
        s = -s;
        qc(kb - m, 0) = RTHALF * mult.q(km - 1); // Adjust index
        qc(kb - m, 1) = -RTHALF * mult.q(km);    // Adjust index
        qc(kb + m, 0) = s * mult.q(km - 1);      // Adjust index
        qc(kb + m, 1) = s * mult.q(km);          // Adjust index
        km += 2;
      }
    }
  }

  return qc;
}

void MultipoleShifter::shift_multipoles(const Mult &q1, int l1, int m1,
                                        Mult &q2, int m2, const Vec3 &pos) {
  // Constants
  constexpr double RTHALF = 0.7071067811865475244;
  constexpr double eps = 0.0; // No early termination
  const double &x = pos.x();
  const double &y = pos.y();
  const double &z = pos.z();

  // Return if parameters are invalid
  if (l1 > m1 || l1 > m2)
    return;

  // Estimate largest significant transferred multipole
  int N = estimate_largest_transferred_multipole(pos, q1, l1, m1, m2, eps);
  int k1 = std::max(1, l1);
  size_t num_components = (N + 1) * (N + 1) + 1;

  Mat rc = get_cplx_sh(pos, N);
  Mat qc = get_cplx_mults(q1, l1, m1, N);

  // Construct shifted complex multipoles QZ (only for non-negative M)
  Mat qz = Mat::Zero(num_components, 2); // Use dynamic size based on N

  if (l1 == 0) {
    qz(1, 0) = qc(1, 0); // Use index 1 for first element
    qz(1, 1) = qc(1, 1);
  }

  // Create a BinomialCoefficients object for rtbinom calculations
  BinomialCoefficients binomials(20);

  for (int l = k1; l <= N; l++) {
    int kmax = std::min(l, m1);
    int lb = l * l + l + 1;
    int lm = lb;

    for (int m = 0; m <= l; m++) {
      qz(lm, 0) = 0.0;
      qz(lm, 1) = 0.0;

      if (l1 == 0) {
        qz(lm, 0) = qc(1, 0) * rc(lm, 0) - qc(1, 1) * rc(lm, 1);
        qz(lm, 1) = qc(1, 0) * rc(lm, 1) + qc(1, 1) * rc(lm, 0);
      }

      for (int k = k1; k <= kmax; k++) {
        int qmin = std::max(-k, k - l + m);
        int qmax = std::min(k, l - k + m);
        int kb = k * k + k + 1;
        int jb = (l - k) * (l - k) + (l - k) + 1;

        for (int qq = qmin; qq <= qmax; qq++) {
          double factor = binomials.sqrt_binomial(l + m, k + qq) *
                          binomials.sqrt_binomial(l - m, k - qq);

          qz(lm, 0) += factor * (qc(kb + qq, 0) * rc(jb + m - qq, 0) -
                                 qc(kb + qq, 1) * rc(jb + m - qq, 1));
          qz(lm, 1) += factor * (qc(kb + qq, 0) * rc(jb + m - qq, 1) +
                                 qc(kb + qq, 1) * rc(jb + m - qq, 0));
        }
      }

      lm++;
    }
  }

  // Construct real multipoles and add to Q2
  if (l1 == 0)
    q2.q(0) += qz(1, 0); // Map from temp array index 1 to q2 index 0

  for (int k = k1; k <= N; k++) {
    int kb = k * k + k + 1;
    int km = k * k + 1;

    q2.q(km - 1) += qz(kb, 0); // Adjust index when writing to q2

    double s = 1.0 / RTHALF;
    km++;

    for (int m = 1; m <= k; m++) {
      s = -s;
      q2.q(km - 1) += s * qz(kb + m, 0); // Adjust index when writing to q2
      q2.q(km) += s * qz(kb + m, 1);     // Adjust index when writing to q2
      km += 2;
    }
  }
}

} // namespace occ::dma
