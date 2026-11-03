#pragma once
#include <array>
#include <cmath>
#include <occ/numint/grid_types.h>

#include <utility>
#include <vector>

namespace occ::numint {

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
 * @brief Grid-weight response: sum_g e_g dw_g/dR_B for every atom B.
 *
 * w_g = w0_g P_A(r_g) / sum_C P_C(r_g) is the Becke-partitioned weight of a
 * point attached to its parent atom A (the point moves with A). This returns
 * the derivative of sum_g e_g w_g with e_g held fixed, 3 x natoms. Only the
 * Becke partition (with Becke or Treutler-Ahlrichs size adjustment, exactly as
 * calculate_atomic_grid_weights builds it) is supported.
 *
 * @param points all grid points (3 x N), grouped by parent atom
 * @param weights the partitioned weights w_g
 * @param atom_blocks (offset, count) of each atom's points
 * @param e per-point quantity e_g (e.g. the XC energy density)
 * @param positions atomic positions (3 x natoms)
 * @param radii atomic radii used by the size adjustment
 * @param treutler_ahlrichs whether the Treutler-Ahlrichs adjustment was used
 */
Mat3N becke_weight_gradient(
    const Mat3N &points, const Vec &weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks, const Vec &e,
    const Mat3N &positions, const Vec &radii, bool treutler_ahlrichs);

/**
 * @brief Per-point derivatives of Becke-partitioned weights.
 *
 * For a point r attached to parent atom A with partitioned weight w,
 * operator() fills dw (3 x natoms) with dw/dR_B for every atom B, the
 * parent's column by translational invariance. Same partition (Becke cell
 * function, Becke or Treutler-Ahlrichs size adjustment) as
 * calculate_atomic_grid_weights.
 */
class BeckeWeightDerivative {
public:
  BeckeWeightDerivative(const Mat3N &positions, const Vec &radii,
                        bool treutler_ahlrichs);
  /// Returns false (dw zero) when the weight vanishes.
  bool operator()(const Vec3 &r, double w, int parent, Mat3N &dw) const;
  /// The same for a batch of points with one parent: row g holds dw_g
  /// (3 x natom, column-major: 3 A + x).
  Mat batch(Eigen::Ref<const Mat3N> points, Eigen::Ref<const Vec> w,
            int parent) const;

private:
  Mat3N m_positions;
  // 1 / R (zero diagonal) and a^T: column C holds row C of 1 / R and a, so
  // a cell's functions s(C, .) are formed from contiguous memory.
  Mat m_invR, m_aT;
  int m_natom{0};
};

/**
 * @brief Second-order grid-weight response: sum_g e_g d2 w_g / dR_X dR_Y
 * (3N x 3N), points moving with their parent atoms, e_g held fixed. Same
 * partition as becke_weight_gradient.
 */
Mat becke_weight_hessian(
    const Mat3N &points, const Vec &weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks, const Vec &e,
    const Mat3N &positions, const Vec &radii, bool treutler_ahlrichs);

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
                                  const Mat &atomic_positions, const Vec &radii,
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
 * @brief Apply ORCA-style 5-region angular pruning
 *
 * ORCA divides the radial grid into 5 regions, each with a different
 * number of angular points. The region boundaries are scaled by the
 * atomic radius and are optimized for each element.
 *
 * @param atomic_number Atomic number
 * @param region_grids Array of 5 Lebedev grid sizes for each region
 * @param radii Radial grid points in Bohr
 * @return IVec Number of angular points for each radial point
 */
IVec prune_orca_scheme(size_t atomic_number,
                       const std::array<size_t, 5> &region_grids,
                       const Vec &radii);

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
 * @brief Perez-Jorda Gauss-Chebyshev (second kind) quadrature with the
 * Treutler-Ahlrichs M3 mapping r = xi / ln 2 * ln(2 / (1 - x)).
 *
 * Nodes x_i = (n+1-2i)/(n+1) + 2/pi (1 + 2/3 sin^2 t_i) cos t_i sin t_i,
 * weights 16/(3(n+1)) sin^4 t_i, t_i = i pi / (n+1) [Perez-Jorda, San-Fabian,
 * Moscardo, Comput. Phys. Commun. 70, 271 (1992)]. This is ORCA's "OptM3 with
 * GC (2021)" radial grid. Weights include r^2 (not 4 pi); points ascending.
 */
RadialGrid generate_perez_jorda_m3_radial_grid(size_t num_points, double xi);

/// One ORCA 6.1 COSX atomic grid: number of radial shells, the M3 scaling
/// xi (bohr), and the angular grid as runs of (Lebedev order, shell count)
/// from the nucleus outwards.
struct OrcaCOSXAtomicGrid {
  int num_radial{0};
  double xi{0.0};
  const std::vector<std::pair<int, int>> *angular_runs{nullptr};
};

/**
 * @brief ORCA 6.1 COSX atomic grid for an element and GRIDX level (1-3).
 *
 * Measured from the grids ORCA 6.1.1 writes for single atoms (exact to
 * 1e-9 in radii and weights). Returns num_radial == 0 for Z > 86.
 */
OrcaCOSXAtomicGrid orca_cosx_atomic_grid(int atomic_number, int level);

/**
 * @brief Generate a Gauss-Chebyshev radial grid
 *
 * @param num_points Number of radial points
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_gauss_chebyshev_radial_grid(size_t num_points);

/**
 * @brief Generate a Gauss-Chebyshev radial grid with M3 mapping
 *
 * Uses the Treutler-Alrichs M3 mapping for transforming the Gauss-Chebyshev
 * quadrature to an infinite interval suitable for atomic integrals:
 *   r = (xi / ln2) * ln(2 / (1-x))
 * where xi is a scaling parameter based on the atomic radius.
 *
 * Reference: O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995)
 *
 * @param num_points Number of radial points
 * @param atomic_number Atomic number (used for xi scaling)
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_gauss_chebyshev_m3_radial_grid(size_t num_points,
                                                    size_t atomic_number);

/**
 * @brief Generate a Euler-Maclaurin radial grid
 *
 * @param num_points Number of radial points
 * @param alpha
 * @return RadialGrid The generated radial grid
 */
RadialGrid generate_euler_maclaurin_radial_grid(size_t num_points,
                                                double alpha);

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
AtomGrid generate_atom_grid(size_t atomic_number, const GridSettings &settings,
                            RadialGridMethod method = RadialGridMethod::LMG,
                            double alpha_max = 0.0, int l_max = 0,
                            const Vec &alpha_min = Vec());

} // namespace occ::numint
