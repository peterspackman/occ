#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <vector>

namespace occ::dma {

/**
 * @brief Handles 1D multipole shifting operations along a single axis
 *
 * This class encapsulates the functionality for shifting multipole moments
 * along a linear axis (typically z-axis) and moving them between sites.
 * It provides a modern C++ interface for operations that were originally
 * implemented as separate shiftz/movez functions.
 */
class LinearMultipoleShifter {
public:
  /**
   * @brief Construct a shifter for moving multipoles to sites along an axis
   *
   * @param position Source position along the axis
   * @param multipoles Source multipoles to be moved
   * @param site_positions Matrix of site positions (only z-component used)
   * @param site_radii Vector of site radii
   * @param site_limits Vector of maximum multipole ranks for each site
   * @param site_multipoles Vector of multipoles at each site
   * @param max_rank Maximum multipole rank to consider
   */
  LinearMultipoleShifter(double position, Mult &multipoles,
                         const Mat3N &site_positions, const Vec &site_radii,
                         const IVec &site_limits,
                         std::vector<Mult> &site_multipoles, int max_rank);

  /**
   * @brief Move multipoles from source position to nearest appropriate sites
   *
   * This implements the logic from the original movez function, distributing
   * multipoles to the nearest sites along the axis based on distance and
   * site limits.
   */
  void move_to_sites();

  /**
   * @brief Shift multipoles along the axis between two points
   *
   * @param source Source multipoles
   * @param l1 Minimum rank to shift
   * @param m1 Maximum rank to shift from source
   * @param destination Destination multipoles
   * @param m2 Maximum rank to keep at destination
   * @param displacement Displacement along the axis
   */
  static void shift_along_axis(const Mult &source, int l1, int m1,
                               Mult &destination, int m2, double displacement);

private:
  /**
   * @brief Find nearest site with sufficient multipole limit
   *
   * @param low Minimum required limit
   * @param start Starting site index for search
   * @return Index of nearest suitable site
   */
  int find_nearest_site_with_limit(int low, int start) const;

  /**
   * @brief Find all sites at approximately the same distance
   *
   * @param primary_site Index of the primary (nearest) site
   * @param low Minimum required limit
   * @param tolerance Distance tolerance for considering sites equivalent
   * @return Vector of site indices at similar distances
   */
  std::vector<int> find_equivalent_sites(int primary_site, int low,
                                         double tolerance = 1.0e-8) const;

  /**
   * @brief Process multipoles for a specific rank range
   *
   * @param sites Vector of destination site indices
   * @param low Starting rank
   * @param high Ending rank
   */
  void process_rank_range(const std::vector<int> &sites, int low, int high);

  /**
   * @brief Calculate scaled distance to a site
   *
   * @param site_index Index of the site
   * @return Scaled distance (distance / radius)
   */
  double scaled_distance_to_site(int site_index) const;

  /**
   * @brief Precompute powers of displacement for efficient shifting
   *
   * @param displacement The displacement value
   * @param max_power Maximum power needed
   * @return Vector of powers [1, z, z^2, z^3, ...]
   */
  static Vec compute_displacement_powers(double displacement, int max_power);

  // Member variables
  double m_position;
  Mult &m_multipoles;
  const Mat3N &m_site_positions;
  const Vec &m_site_radii;
  const IVec &m_site_limits;
  std::vector<Mult> &m_site_multipoles;
  int m_max_rank;
  int m_num_sites;
  int m_site_with_highest_limit;

  // Cached calculations
  Vec m_scaled_distances;
};

/**
 * @brief Convenience function for 1D multipole shifting
 *
 * @param source Source multipoles
 * @param l1 Minimum rank to shift
 * @param m1 Maximum rank to shift from source
 * @param destination Destination multipoles
 * @param m2 Maximum rank to keep at destination
 * @param displacement Displacement along the axis
 */
void shiftz(const Mult &source, int l1, int m1, Mult &destination, int m2,
            double displacement);

/**
 * @brief Convenience function for moving multipoles to sites along an axis
 *
 * @param multipoles Multipole moments to be moved
 * @param position Source position along the axis
 * @param site_positions Matrix of site positions
 * @param site_radii Vector of site radii
 * @param site_limits Vector of maximum rank for each site
 * @param site_multipoles Vector of multipoles at each site
 * @param max_rank Maximum multipole rank
 */
void movez(Mult &multipoles, double position, const Mat3N &site_positions,
           const Vec &site_radii, const IVec &site_limits,
           std::vector<Mult> &site_multipoles, int max_rank);

} // namespace occ::dma