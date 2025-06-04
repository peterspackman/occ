#include <occ/dma/linear_multipole_shifter.h>
#include <occ/core/log.h>
#include <algorithm>
#include <cmath>

namespace occ::dma {

LinearMultipoleShifter::LinearMultipoleShifter(
    double position, Mult &multipoles, const Mat3N &site_positions,
    const Vec &site_radii, const IVec &site_limits,
    std::vector<Mult> &site_multipoles, int max_rank)
    : m_position(position), m_multipoles(multipoles),
      m_site_positions(site_positions), m_site_radii(site_radii),
      m_site_limits(site_limits), m_site_multipoles(site_multipoles),
      m_max_rank(max_rank), m_num_sites(site_positions.cols()),
      m_site_with_highest_limit(0), m_scaled_distances(m_num_sites) {

  // Find site with highest limit
  for (int i = 0; i < m_num_sites; i++) {
    if (m_site_limits(i) > m_site_limits(m_site_with_highest_limit)) {
      m_site_with_highest_limit = i;
    }
  }

  // Precompute scaled distances to all sites
  for (int i = 0; i < m_num_sites; i++) {
    m_scaled_distances(i) = scaled_distance_to_site(i);
  }
}

void LinearMultipoleShifter::move_to_sites() {
  int low = 0;

  while (true) {
    // Find nearest site with sufficient limit
    int primary_site = find_nearest_site_with_limit(low, m_site_with_highest_limit);
    
    // Find all sites at approximately the same distance
    std::vector<int> equivalent_sites = find_equivalent_sites(primary_site, low);
    
    // Process this rank range
    process_rank_range(equivalent_sites, low, m_site_limits(primary_site));
    
    // If we've processed all ranks, we're done
    if (m_site_limits(primary_site) >= m_max_rank) {
      break;
    }
    
    // Move to next rank range
    low = m_site_limits(primary_site) + 1;
  }
}

void LinearMultipoleShifter::shift_along_axis(const Mult &source, int l1, int m1,
                                              Mult &destination, int m2,
                                              double displacement) {
  // Skip if nothing to shift
  if (l1 > m1) {
    return;
  }

  // Precompute powers of displacement for efficiency
  Vec powers = compute_displacement_powers(displacement, m2);

  // Shift charge (rank 0)
  if (l1 == 0) {
    destination.q(0) += source.q(0);
    for (int n = 1; n <= m2; n++) {
      destination.q(n) += source.q(0) * powers(n);
    }
  }

  // Nothing more to do if we only had charge
  if (m1 <= 0) {
    return;
  }

  // Shift higher multipoles
  int s1 = std::max(1, l1);
  int s2 = std::min(m1, m2);

  for (int s = s1; s <= s2; s++) {
    destination.q(s) += source.q(s);

    for (int n = s + 1; n <= m2; n++) {
      // Calculate binomial coefficient efficiently
      double binomial = 1.0;
      for (int k = 0; k < s; k++) {
        binomial *= double(n - k) / double(s - k);
      }
      destination.q(n) += binomial * source.q(s) * powers(n - s);
    }
  }
}

int LinearMultipoleShifter::find_nearest_site_with_limit(int low, int start) const {
  int nearest = start;
  for (int i = 0; i < m_num_sites; i++) {
    if (m_scaled_distances(i) < m_scaled_distances(nearest) && 
        m_site_limits(i) >= low) {
      nearest = i;
    }
  }
  return nearest;
}

std::vector<int> LinearMultipoleShifter::find_equivalent_sites(int primary_site, 
                                                               int low, 
                                                               double tolerance) const {
  std::vector<int> equivalent_sites;
  equivalent_sites.push_back(primary_site);
  
  const double primary_distance = m_scaled_distances(primary_site);
  const int primary_limit = m_site_limits(primary_site);
  
  for (int i = 0; i < m_num_sites; i++) {
    if (i == primary_site) continue;
    if (m_scaled_distances(i) > primary_distance + tolerance) continue;
    if (m_site_limits(i) < low) continue;
    if (m_site_limits(i) != primary_limit) continue;
    
    equivalent_sites.push_back(i);
  }
  
  return equivalent_sites;
}

void LinearMultipoleShifter::process_rank_range(const std::vector<int> &sites, 
                                                int low, int high) {
  const int num_sites = sites.size();
  
  // If multiple sites, split the contribution equally
  if (num_sites > 1) {
    for (int rank = low; rank <= high; rank++) {
      m_multipoles.q(rank) /= num_sites;
    }
  }
  
  // Shift multipoles to each equivalent site
  for (int site_idx : sites) {
    const double displacement = m_position - m_site_positions(2, site_idx);
    shift_along_axis(m_multipoles, low, high, m_site_multipoles[site_idx], 
                     m_max_rank, displacement);
  }
  
  // Zero out the transferred multipoles
  for (int rank = low; rank <= high; rank++) {
    m_multipoles.q(rank) = 0.0;
  }
  
  // If we haven't reached max rank, shift higher multipoles back
  if (high < m_max_rank) {
    for (int site_idx : sites) {
      const double back_displacement = m_site_positions(2, site_idx) - m_position;
      shift_along_axis(m_site_multipoles[site_idx], high + 1, m_max_rank, 
                       m_multipoles, m_max_rank, back_displacement);
      
      // Zero out the transferred multipoles at the site
      for (int rank = high + 1; rank <= m_max_rank; rank++) {
        m_site_multipoles[site_idx].q(rank) = 0.0;
      }
    }
  }
}

double LinearMultipoleShifter::scaled_distance_to_site(int site_index) const {
  return std::abs(m_site_positions(2, site_index) - m_position) / 
         m_site_radii(site_index);
}

Vec LinearMultipoleShifter::compute_displacement_powers(double displacement, 
                                                        int max_power) {
  Vec powers = Vec::Zero(max_power + 1);
  powers(0) = 1.0;
  for (int i = 1; i <= max_power; i++) {
    powers(i) = displacement * powers(i - 1);
  }
  return powers;
}

// Convenience functions that maintain the original interface
void shiftz(const Mult &source, int l1, int m1, Mult &destination, int m2,
            double displacement) {
  LinearMultipoleShifter::shift_along_axis(source, l1, m1, destination, m2, 
                                           displacement);
}

void movez(Mult &multipoles, double position, const Mat3N &site_positions,
           const Vec &site_radii, const IVec &site_limits,
           std::vector<Mult> &site_multipoles, int max_rank) {
  LinearMultipoleShifter shifter(position, multipoles, site_positions, 
                                 site_radii, site_limits, site_multipoles, 
                                 max_rank);
  shifter.move_to_sites();
}

} // namespace occ::dma