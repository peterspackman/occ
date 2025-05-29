#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/qm/wavefunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace occ::dma {

/**
 * @brief Settings for linear multipole analysis
 */
struct LinearDMASettings {
  int max_rank = 4;
  bool include_nuclei = true;
  bool use_slices = false;
  double tolerance = 2.30258 * 18; // Threshold for numerical significance
  double default_radius = 0.5;     // Default site radius in Å
  double hydrogen_radius = 0.325;  // Hydrogen site radius in Å
};

/**
 * @brief Calculator for distributed multipole analysis of linear molecules
 *
 * This class encapsulates the functionality of the dmaql0 function,
 * providing a cleaner interface for calculating multipole moments
 * for linear molecules where only Qlm with m=0 are non-zero.
 */
class LinearMultipoleCalculator {
public:
  /**
   * @brief Constructor for linear multipole calculator
   *
   * @param wfn Wavefunction containing basis set and density matrix
   * @param settings Settings for the DMA calculation
   */
  LinearMultipoleCalculator(
      const occ::qm::Wavefunction &wfn,
      const LinearDMASettings &settings = LinearDMASettings{});

  /**
   * @brief Calculate all multipole moments for the linear molecule
   *
   * @return std::vector<Mult> Multipole moments for each site
   */
  std::vector<Mult> calculate();

private:
  /**
   * @brief Setup sites and their properties based on molecule atoms
   */
  void setup_sites();

  /**
   * @brief Setup slice information for slice-based calculations
   */
  void setup_slices();

  /**
   * @brief Process nuclear contributions to multipoles
   *
   * @param site_multipoles Vector to store multipole contributions
   */
  void process_nuclear_contributions(std::vector<Mult> &site_multipoles);

  /**
   * @brief Process electronic contributions from density matrix
   *
   * @param site_multipoles Vector to store multipole contributions
   */
  void process_electronic_contributions(std::vector<Mult> &site_multipoles);

  /**
   * @brief Process a shell pair for electronic contributions
   *
   * @param shell_i First shell
   * @param shell_j Second shell
   * @param i_shell_idx Index of first shell
   * @param j_shell_idx Index of second shell
   * @param atom_i Index of first atom
   * @param atom_j Index of second atom
   * @param site_multipoles Vector to store multipole contributions
   */
  void process_shell_pair(const occ::qm::Shell &shell_i,
                          const occ::qm::Shell &shell_j, int i_shell_idx,
                          int j_shell_idx, int atom_i, int atom_j,
                          std::vector<Mult> &site_multipoles);

  /**
   * @brief Process a primitive pair within shells
   *
   * @param shell_i First shell
   * @param shell_j Second shell
   * @param i_prim Index of primitive in first shell
   * @param j_prim Index of primitive in second shell
   * @param d_block Density matrix block for this shell pair
   * @param atom_i Index of first atom
   * @param atom_j Index of second atom
   * @param site_multipoles Vector to store multipole contributions
   */
  void process_primitive_pair(const occ::qm::Shell &shell_i,
                              const occ::qm::Shell &shell_j, int i_prim,
                              int j_prim, const Mat &d_block, int atom_i,
                              int atom_j, std::vector<Mult> &site_multipoles);

  /**
   * @brief Calculate error function integrals for slices
   *
   * @param aa Sum of gaussian exponents
   * @param la Angular momentum of first function
   * @param lb Angular momentum of second function
   * @param za Z-coordinate of first center relative to product center
   * @param zb Z-coordinate of second center relative to product center
   * @param z1 Lower bound of slice
   * @param z2 Upper bound of slice
   * @param gz Output tensor for integrals
   * @param skip Flag set to true if all integrals are negligible
   */
  void calculate_slice_integrals(double aa, int la, int lb, double za,
                                 double zb, double z1, double z2,
                                 Eigen::Tensor<double, 3> &gz,
                                 bool &skip) const;

  /**
   * @brief Get cartesian powers for a basis function
   *
   * @param bf_idx Basis function index within shell
   * @param l Angular momentum of shell
   * @param powers Output array for x,y,z powers
   */
  void get_cartesian_powers(int bf_idx, int l, int powers[3]) const;

  /**
   * @brief Apply scaling factors for higher angular momentum functions
   *
   * @param d_block Density matrix block to scale
   * @param l Angular momentum
   * @param size Size of shell
   * @param is_row Whether to scale rows (true) or columns (false)
   */
  void apply_angular_scaling(Mat &d_block, int l, int size, bool is_row) const;

  // Member variables
  const occ::qm::Wavefunction &m_wfn;
  const LinearDMASettings m_settings;

  // Cached data from wavefunction
  Mat3N m_sites;      // Site positions (same as atom positions initially)
  Vec m_site_radii;   // Site radii
  IVec m_site_limits; // Site multipole rank limits
  std::vector<int> m_sort_indices; // Sorting indices for z-ordering
  Vec m_slice_separations;         // Separation planes for slices

  // Density matrix (factor of 2 applied)
  Mat m_density_matrix;
};

} // namespace occ::dma
