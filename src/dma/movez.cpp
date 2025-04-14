#include <occ/core/log.h>
#include <occ/dma/movez.h>

namespace occ::dma {

void shiftz(const Mult &q1, int l1, int m1, Mult &q2, int m2, double z) {
  // Skip if nothing to shift
  if (l1 > m1)
    return;

  // Precompute powers of z
  Vec zs = Vec::Zero(m2 + 1);
  zs(0) = 1.0;
  for (int i = 1; i <= m2; i++) {
    zs(i) = z * zs(i - 1);
  }

  // Shift charge (rank 0)
  if (l1 == 0) {
    q2.q(0) += q1.q(0);
    for (int n = 1; n <= m2; n++) {
      q2.q(n) += q1.q(0) * zs(n);
    }
  }

  // Nothing more to do if we only had charge
  if (m1 <= 0)
    return;

  // Shift higher multipoles
  int s1 = std::max(1, l1);
  int s2 = std::min(m1, m2);

  for (int s = s1; s <= s2; s++) {
    q2.q(s) += q1.q(s);

    for (int n = s + 1; n <= m2; n++) {
      // Use binomial coefficients for shifting
      double binomial = 1.0;
      for (int k = 0; k < s; k++) {
        binomial *= double(n - k) / double(s - k);
      }
      q2.q(n) += binomial * q1.q(s) * zs(n - s);
    }
  }
}

void movez(Mult &qp, double p, const Mat3N &sites, const Vec &site_radii,
           const IVec &site_limits, std::vector<Mult> &site_multipoles,
           int max_rank) {
  const double eps = 1.0e-8;
  const int n_sites = sites.cols();

  // Calculate scaled distances to all sites
  Vec r(n_sites);
  for (int i = 0; i < n_sites; i++) {
    r(i) = std::abs(sites(2, i) - p) / site_radii(i);
  }

  // Find site with largest limit
  int j = 0;
  for (int i = 0; i < n_sites; i++) {
    if (site_limits(i) > site_limits(j))
      j = i;
  }

  // Start with lowest rank multipoles
  int low = 0;

  while (true) {
    // Find nearest site with sufficient limit
    int k = j;
    for (int i = 0; i < n_sites; i++) {
      if (r(i) < r(k) && site_limits(i) >= low)
        k = i;
    }

    // Find all sites at same distance (within tolerance)
    int n = 1;
    std::vector<int> m(2, k); // Start with nearest site

    for (int i = 0; i < n_sites; i++) {
      if (i == k)
        continue;
      if (r(i) > r(k) + eps)
        continue;
      if (site_limits(i) < low)
        continue;
      if (site_limits(i) != site_limits(k))
        continue;

      if (n < 2) {
        m[n] = i;
        n++;
      }
    }

    // Log actions if verbose
    log::info("From {:.3f}: ranks {} to {} to be moved to site at {:.3f}", p,
              low, site_limits(k), sites(2, m[0]));
    if (n == 2) {
      log::info("and site {} at {:.3f}", m[1], sites(2, m[1]));
    }

    // If multiple sites at same distance, split contribution
    if (n == 2) {
      for (int i = low; i <= site_limits(k); i++) {
        qp.q(i) *= 0.5;
      }
    }

    // Move multipoles to each selected site
    for (int i = 0; i < n; i++) {
      k = m[i];
      shiftz(qp, low, site_limits(k), site_multipoles[k], max_rank,
             p - sites(2, k));
    }

    // Zero out moved multipoles
    for (int i = low; i <= site_limits(k); i++) {
      qp.q(i) = 0.0;
    }

    // If all multipoles moved, we're done
    if (site_limits(k) >= max_rank)
      return;

    // Otherwise, shift higher rank multipoles back to original center
    // and continue with next rank
    for (int i = 0; i < n; i++) {
      k = m[i];
      shiftz(site_multipoles[k], site_limits(k) + 1, max_rank, qp, max_rank,
             sites(2, k) - p);

      // Zero out shifted multipoles
      for (int l = site_limits(k) + 1; l <= max_rank; l++) {
        site_multipoles[k].q(l) = 0.0;
      }
    }

    // Update lower bound for next iteration
    low = site_limits(k) + 1;
  }
}
} // namespace occ::dma
