#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/multipole.h>
#include <occ/numint/molecular_grid.h>
#include <occ/qm/mo.h>
#include <occ/gto/shell.h>
#include <occ/slater/slaterbasis.h>
#include <vector>

namespace occ::dft {

/**
 * @brief Class implementing Hirshfeld atomic charge analysis
 *
 * Hirshfeld charges are computed by partitioning the molecular electron density
 * at each grid point according to the ratio of the promolecular density
 * (superposition of atomic densities) for each atom to the total promolecular
 * density.
 *
 * This class can also compute higher-order atomic multipoles (dipoles,
 * quadrupoles, octupoles and hexadecapoles) using the same Hirshfeld
 * partitioning approach.
 */
class HirshfeldPartition {
public:
  /**
   * @brief Constructor for Hirshfeld charge analysis
   *
   * @param basis Basis set for the molecule
   * @param max_multipole_order Maximum order of multipoles to compute (0-4)
   * @param charge Total molecular charge
   */
  HirshfeldPartition(const occ::gto::AOBasis &basis, int max_multipole_order = 0,
                     int charge = 0);

  /**
   * @brief Calculate Hirshfeld charges for a given set of molecular orbitals
   *
   * @param mo Molecular orbitals containing the density matrix
   * @return Vec Vector of Hirshfeld charges for each atom
   */
  Vec calculate(const occ::qm::MolecularOrbitals &mo);

  /**
   * @brief Calculate Hirshfeld multipoles for a given set of molecular orbitals
   *
   * @param mo Molecular orbitals containing the density matrix
   * @return std::vector<occ::core::Multipole<4>> Vector of Hirshfeld multipoles
   * for each atom
   */
  std::vector<occ::core::Multipole<4>>
  calculate_multipoles(const occ::qm::MolecularOrbitals &mo);

  /**
   * @brief Get the last calculated Hirshfeld charges
   *
   * @return const Vec& Reference to the Hirshfeld charges
   */
  inline const Vec &charges() const { return m_hirshfeld_charges; }

  /**
   * @brief Get the last calculated Hirshfeld multipoles
   *
   * @return const std::vector<occ::core::Multipole<4>>& Reference to the
   * Hirshfeld multipoles
   */
  inline const std::vector<occ::core::Multipole<4>> &multipoles() const {
    return m_multipoles;
  }

  /**
   * @brief Get the volumes of each atom in the molecule
   *
   * @return const Vec& Reference to the atomic volumes
   */
  inline const Vec &atom_volumes() const { return m_atom_volumes; }

  /**
   * @brief Get the volumes of each free atom
   *
   * @return const Vec& Reference to the free atom volumes
   */
  inline const Vec &free_atom_volumes() const { return m_free_atom_volumes; }

private:
  /**
   * @brief Calculate Hirshfeld weights, charges, and multipoles
   *
   * @param mo Molecular orbitals containing the density matrix
   * @param calculate_higher_multipoles Whether to calculate multipoles beyond
   * charges
   */
  void compute_hirshfeld_weights(const occ::qm::MolecularOrbitals &mo,
                                 bool calculate_higher_multipoles = false);

  occ::gto::AOBasis m_basis;
  occ::dft::MolecularGrid m_grid;
  std::vector<occ::dft::AtomGrid> m_atom_grids;
  std::vector<occ::slater::Basis> m_slater_basis;
  Mat m_density_matrix;
  Vec m_hirshfeld_charges;
  std::vector<occ::core::Multipole<4>> m_multipoles;
  Vec m_atom_volumes;
  Vec m_free_atom_volumes;
  bool m_atomic_ion{false};
  int m_charge{0};
  int m_max_multipole_order{0};
};

/**
 * @brief Calculate Hirshfeld charges for a molecule
 *
 * Convenience function to compute Hirshfeld charges in a single call.
 *
 * @param basis Basis set for the molecule
 * @param mo Molecular orbitals containing the density matrix
 * @param charge Total molecular charge
 * @return Vec Vector of Hirshfeld charges for each atom
 */
Vec calculate_hirshfeld_charges(const occ::gto::AOBasis &basis,
                                const occ::qm::MolecularOrbitals &mo,
                                int charge = 0);

/**
 * @brief Calculate Hirshfeld multipoles for a molecule
 *
 * Convenience function to compute Hirshfeld multipoles in a single call.
 *
 * @param basis Basis set for the molecule
 * @param mo Molecular orbitals containing the density matrix
 * @param max_multipole_order Maximum order of multipoles to compute (0-4)
 * @param charge Total molecular charge
 * @return std::vector<occ::core::Multipole<4>> Vector of Hirshfeld multipoles
 * for each atom
 */
std::vector<occ::core::Multipole<4>>
calculate_hirshfeld_multipoles(const occ::gto::AOBasis &basis,
                               const occ::qm::MolecularOrbitals &mo,
                               int max_multipole_order = 4, int charge = 0);

namespace impl {

/**
 * @brief Kernel function for calculating Hirshfeld weights with restricted
 * orbitals
 *
 * @param r Distance matrix from grid points to atoms
 * @param rho Electron density and derivatives at grid points
 * @param weights Grid weights
 * @param rho_pro Promolecular density at grid points
 * @param hirshfeld_charges Output: Hirshfeld charges
 * @param atom_volumes Output: Atomic volumes
 * @param num_electrons Output: Total number of electrons
 * @param num_electrons_promol Output: Total number of electrons in promolecule
 */
void hirshfeld_kernel_restricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Vec> atom_volumes,
    double &num_electrons, double &num_electrons_promol);

/**
 * @brief Kernel function for calculating Hirshfeld weights with unrestricted
 * orbitals
 *
 * @param r Distance matrix from grid points to atoms
 * @param rho Electron density and derivatives at grid points
 * @param weights Grid weights
 * @param rho_pro Promolecular density at grid points
 * @param hirshfeld_charges Output: Hirshfeld charges
 * @param atom_volumes Output: Atomic volumes
 * @param num_electrons Output: Total number of electrons
 * @param num_electrons_promol Output: Total number of electrons in promolecule
 */
void hirshfeld_kernel_unrestricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Vec> atom_volumes,
    double &num_electrons, double &num_electrons_promol);

/**
 * @brief Kernel function for calculating Hirshfeld multipoles with restricted
 * orbitals
 *
 * @param r Distance matrix from grid points to atoms
 * @param r_vec Vectors from atom centers to grid points (for each atom)
 * @param rho Electron density at grid points
 * @param weights Grid weights
 * @param rho_pro Promolecular density at grid points
 * @param multipoles Output: Hirshfeld multipoles
 * @param atom_volumes Output: Atomic volumes
 * @param num_electrons Output: Total number of electrons
 * @param num_electrons_promol Output: Total number of electrons in promolecule
 * @param max_multipole_order Maximum order of multipoles to compute
 */
void hirshfeld_multipole_kernel_restricted(
    Eigen::Ref<const Mat> r, const std::vector<Mat3N> &r_vec,
    Eigen::Ref<const Mat> rho, Eigen::Ref<const Vec> weights,
    Eigen::Ref<const Mat> rho_pro,
    std::vector<occ::core::Multipole<4>> &multipoles,
    Eigen::Ref<Vec> atom_volumes, double &num_electrons,
    double &num_electrons_promol, int max_multipole_order);

/**
 * @brief Kernel function for calculating Hirshfeld multipoles with unrestricted
 * orbitals
 *
 * @param r Distance matrix from grid points to atoms
 * @param r_vec Vectors from atom centers to grid points (for each atom)
 * @param rho Electron density at grid points
 * @param weights Grid weights
 * @param rho_pro Promolecular density at grid points
 * @param multipoles Output: Hirshfeld multipoles
 * @param atom_volumes Output: Atomic volumes
 * @param num_electrons Output: Total number of electrons
 * @param num_electrons_promol Output: Total number of electrons in promolecule
 * @param max_multipole_order Maximum order of multipoles to compute
 */
void hirshfeld_multipole_kernel_unrestricted(
    Eigen::Ref<const Mat> r, const std::vector<Mat3N> &r_vec,
    Eigen::Ref<const Mat> rho, Eigen::Ref<const Vec> weights,
    Eigen::Ref<const Mat> rho_pro,
    std::vector<occ::core::Multipole<4>> &multipoles,
    Eigen::Ref<Vec> atom_volumes, double &num_electrons,
    double &num_electrons_promol, int max_multipole_order);

} // namespace impl

} // namespace occ::dft
