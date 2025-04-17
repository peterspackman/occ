#pragma once
#include <array>
#include <cmath>
#include <occ/dft/grid_types.h>

namespace occ::dft {

/**
 * @brief Array of available Lebedev grid levels
 *
 * Each element in the array represents a valid number of Lebedev grid points.
 */
extern const std::array<uint_fast16_t, 33> lebedev_grid_levels;

/**
 * @brief Find the nearest Lebedev grid level at or above the requested number
 * of points
 *
 * @param n Requested number of angular points
 * @return uint_fast16_t The actual number of angular points in the Lebedev grid
 */
uint_fast16_t nearest_grid_level_at_or_above(uint_fast16_t n);

/**
 * @brief Find the nearest Lebedev grid level below the requested number of
 * points
 *
 * @param n Requested number of angular points
 * @return uint_fast16_t The actual number of angular points in the Lebedev grid
 */
uint_fast16_t nearest_grid_level_below(uint_fast16_t n);

/**
 * @brief Get the number of points in a Lebedev grid
 *
 * @param level The Lebedev grid level
 * @return size_t The number of points in the grid
 */
size_t angular_point_count(size_t level);

/**
 * @brief Get the atomic radius for a given atomic number
 *
 * @param atomic_number The atomic number
 * @return double The atomic radius in Bohr
 */
double get_atomic_radius(size_t atomic_number);

/**
 * @brief Apply the Becke partition function
 *
 * @param w Input values to partition
 * @return Vec Partitioned values
 */
Vec becke_partition(const Vec &w);

/**
 * @brief Apply the Stratmann-Scuseria partition function
 *
 * @param w Input values to partition
 * @return Vec Partitioned values
 */
Vec stratmann_scuseria_partition(const Vec &w);

/**
 * @brief Calculate interatomic distances
 *
 * @param positions Atomic positions (3 x N)
 * @return Mat Matrix of interatomic distances
 */
Mat calculate_interatomic_distances(const Mat3N &positions);

/**
 * @brief Calculate Becke atomic weights for grid points
 *
 * @param grid_points Grid points (3 x N)
 * @param atomic_positions Atomic positions (3 x M)
 * @param radii atomic radii (M)
 * @param interatomic_distances Matrix of interatomic distances (M x M)
 * @return Mat Matrix of weights (N x M)
 */
Mat calculate_atomic_grid_weights(PartitionMethod method,
                                  const Mat &grid_points,
                                  const Mat &atomic_positions,
                                  const Vec &radii,
                                  const Mat &interatomic_distances);

/**
 * @brief Apply the NWChem pruning scheme to a radial grid
 *
 * @param nuclear_charge Atomic number
 * @param max_angular Maximum number of angular points
 * @param num_radial Number of radial points
 * @param radii Radial grid points
 * @return IVec Number of angular points for each radial point
 */
IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular,
                         size_t num_radial, const Vec &radii);

/**
 * @brief Apply the NumGrid pruning scheme to a radial grid
 *
 * @param atomic_number Atomic number
 * @param max_angular Maximum number of angular points
 * @param min_angular Minimum number of angular points
 * @param radii Radial grid points
 * @return IVec Number of angular points for each radial point
 */
IVec prune_numgrid_scheme(size_t atomic_number, size_t max_angular,
                          size_t min_angular, const Vec &radii);

/**
 * @brief Generate a Becke radial grid
 *
 * Reference: A. D. Becke, J. Chem. Phys. 88, 2547 (1988)
 *
 * @param num_points Number of radial points
 * @param rm Scaling parameter
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_becke_radial_grid(size_t num_points, double rm = 1.0);

/**
 * @brief Generate a Mura-Knowles radial grid
 *
 * Reference: M. E. Mura and P. J. Knowles, J. Chem. Phys. 104, 9848 (1996)
 *
 * @param num_points Number of radial points
 * @param charge Atomic number
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_mura_knowles_radial_grid(size_t num_points, size_t charge);

/**
 * @brief Generate a Treutler-Alrichs radial grid
 *
 * Reference: O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995)
 *
 * @param num_points Number of radial points
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_treutler_alrichs_radial_grid(size_t num_points);

/**
 * @brief Generate a Gauss-Chebyshev radial grid
 *
 * @param num_points Number of radial points
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_gauss_chebyshev_radial_grid(size_t num_points);

/**
 * @brief Generate a Euler-Maclaurin radial grid
 *
 * @param num_points Number of radial points
 * @param alpha 
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_euler_maclaurin_radial_grid(size_t num_points, double alpha);


/**
 * @brief Generate a Lindh-Malmqvist-Gagliardi (LMG) radial grid
 *
 * Reference: T. Helgaker, P. Jørgensen, J. Olsen,
 * "Molecular Electronic Structure Theory", John Wiley & Sons, 2000
 *
 * @param atomic_number Atomic number
 * @param radial_precision Precision parameter for the radial grid
 * @param alpha_max Maximum exponent
 * @param l_max Maximum angular momentum
 * @param alpha_min Minimum exponents for each angular momentum
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_lmg_radial_grid(size_t atomic_number,
                                    double radial_precision, double alpha_max,
                                    int l_max, const occ::Vec &alpha_min);

/**
 * @brief Helper function to calculate LMG inner radial boundary
 *
 * @param max_error Maximum error parameter
 * @param alpha_inner Inner alpha parameter
 * @return double Inner radial boundary
 */
double lmg_inner(const double max_error, const double alpha_inner);

/**
 * @brief Helper function to calculate LMG outer radial boundary
 *
 * @param max_error Maximum error parameter
 * @param alpha_outer Outer alpha parameter
 * @param l Angular momentum
 * @param guess Initial guess for the boundary
 * @return double Outer radial boundary
 */
double lmg_outer(const double max_error, const double alpha_outer, const int l,
                 const double guess);

/**
 * @brief Helper function to calculate LMG grid spacing parameter
 *
 * @param max_error Maximum error parameter
 * @param l Angular momentum
 * @param guess Initial guess for the spacing
 * @return double Grid spacing parameter
 */
double lmg_h(const double max_error, const int l, const double guess);

/**
 * @brief Generate an atom-centered grid using the specified method
 * 
 * @param atomic_number Atomic number of the center atom
 * @param settings Grid generation settings
 * @param method The radial grid method to use
 * @param alpha_max Maximum exponent from basis set (optional)
 * @param l_max Maximum angular momentum from basis set (optional)
 * @param alpha_min Minimum exponents for each angular momentum (optional)
 * @return AtomGrid The generated atom-centered grid
 */
AtomGrid generate_atom_grid(
    size_t atomic_number, 
    const GridSettings& settings,
    RadialGridMethod method = RadialGridMethod::LMG,
    double alpha_max = 0.0,
    int l_max = 0,
    const Vec& alpha_min = Vec());

} // namespace occ::dft
