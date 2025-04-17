#include <cmath>
#include <occ/dma/add_qlm.h>
#include <occ/dma/moveq.h>
#include <occ/dma/shiftq.h>

namespace occ::dma {

void moveq(Eigen::Ref<const Vec3> pos, Mult &qt, const Mat3N &site_positions,
           const Vec &site_radii, const IVec &site_limits, std::vector<Mult> &q,
           int lmax) {
  // Constants
  constexpr double eps = 1e-6;
  const double &x = pos.x();
  const double &y = pos.y();
  const double &z = pos.z();

  // Calculate squared distances to each site, normalized by site radius
  const int ns = site_positions.cols();
  Vec rr(ns);

  // Find site with highest limit to start
  int j = 0;
  for (int i = 0; i < ns; i++) {
    // Calculate squared distance normalized by site radius squared
    rr(i) = (pos - site_positions.col(i)).squaredNorm() /
            (site_radii(i) * site_radii(i));

    // Find site with highest limit
    if (site_limits(i) > site_limits(j)) {
      j = i;
    }
  }

  // Calculate (lmax+1)^2 for later use
  int lp1sq = (lmax + 1) * (lmax + 1);

  // Start with lowest rank
  int low = 0;

  // Main loop to process multipoles
  while (true) {
    // Find nearest site with limit >= low
    int k = j;
    for (int i = 0; i < ns; i++) {
      if (rr(i) < rr(k) && site_limits(i) >= low) {
        k = i;
      }
    }

    // Calculate index bounds
    int t1 = low * low;
    int t2 = (site_limits(k) + 1) * (site_limits(k) + 1);

    // If very close to a site, add all multipoles directly
    if (rr(k) <= eps) {

      // Direct transfer of multipoles
      for (int i = t1; i < t2; i++) {
        q[k].q(i) += qt.q(i);
        qt.q(i) = 0.0;
      }

      // If we've reached lmax at this site, we're done
      if (site_limits(k) >= lmax) {
        return;
      }
    } else {
      // Find all sites at approximately the same distance
      std::vector<int> m{k};

      for (int i = 0; i < ns; i++) {
        if (i == k || rr(i) > rr(k) + eps || site_limits(i) != site_limits(k) ||
            site_limits(i) < low) {
          continue;
        }
        m.push_back(i);
      }

      // If multiple equidistant sites, distribute equally
      if (m.size() > 1) {
        double an = 1.0 / m.size();
        for (int i = 0; i < qt.q.size(); i++) {
          qt.q(i) *= an;
        }
      }

      // Shift multipoles to each site
      for (int site_idx : m) {
        // Call shiftq to transfer multipoles
        shiftq(qt, low, site_limits(site_idx), q[site_idx], lmax,
               pos - site_positions.col(site_idx));
      }

      // Zero out the transferred multipoles
      for (int i = t1; i < t2; i++) {
        qt.q(i) = 0.0;
      }

      // If we've reached lmax at this site, we're done
      if (site_limits(k) >= lmax) {
        return;
      }

      // Transfer higher-rank multipoles back to qt
      t1 = t2;
      for (int site_idx : m) {

        // Call shiftq to transfer higher-rank multipoles back to qt
        shiftq(q[site_idx], site_limits(site_idx) + 1, lmax, qt, lmax,
               site_positions.col(site_idx) - pos);

        // Zero out transferred multipoles at the site
        for (int l = t1; l < lp1sq; l++) {
          q[site_idx].q(l) = 0.0;
        }
      }
    }

    // Move to next rank
    low = site_limits(k) + 1;
  }
}
} // namespace occ::dma
