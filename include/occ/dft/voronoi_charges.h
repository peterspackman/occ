#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/numint/molecular_grid.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/mo.h>
#include <occ/gto/shell.h>
#include <occ/slater/slaterbasis.h>
#include <vector>

namespace occ::dft {

using occ::io::GridQuality;

/**
 * @brief Class implementing Voronoi atomic charge analysis using LogSumExp
 * approximation
 *
 * Voronoi charges are computed by partitioning the molecular electron density
 * using a smooth approximation to Voronoi cells via the LogSumExp function.
 * This provides a more robust alternative to traditional Hirshfeld
 * partitioning.
 */
class VoronoiPartition {
public:
  /**
   * @brief Constructor for Voronoi charge analysis
   *
   * @param basis Basis set for the molecule
   * @param charge Total molecular charge
   * @param temperature Temperature parameter for LogSumExp smoothing (default:
   * 0.1; use 0.37 for VDW-scaled)
   * @param use_vdw_radii Whether to scale distances by VDW radii (default:
   * false)
   * @param grid_settings Grid settings for integration (default: Coarse
   * quality)
   */
  VoronoiPartition(
      const occ::gto::AOBasis &basis, int charge = 0, double temperature = 0.1,
      bool use_vdw_radii = false,
      const occ::io::GridSettings &grid_settings =
          occ::io::GridSettings::from_grid_quality(GridQuality::Coarse));

  /**
   * @brief Calculate Voronoi charges for a given set of molecular orbitals
   *
   * @param mo Molecular orbitals containing the density matrix
   * @return Vec Vector of Voronoi charges for each atom
   */
  Vec calculate(const occ::qm::MolecularOrbitals &mo);

  /**
   * @brief Get the last calculated Voronoi charges
   *
   * @return const Vec& Reference to the Voronoi charges
   */
  inline const Vec &charges() const { return m_voronoi_charges; }

  /**
   * @brief Get the volumes of each atom in the molecule
   *
   * @return const Vec& Reference to the atomic volumes
   */
  inline const Vec &atom_volumes() const { return m_atom_volumes; }

  /**
   * @brief Set the temperature parameter for LogSumExp smoothing
   *
   * @param temperature Temperature parameter (smaller values = sharper
   * boundaries)
   */
  void set_temperature(double temperature) { m_temperature = temperature; }

private:
  /**
   * @brief Calculate Voronoi weights and charges using LogSumExp partitioning
   *
   * @param mo Molecular orbitals containing the density matrix
   */
  void compute_voronoi_weights(const occ::qm::MolecularOrbitals &mo);

  /**
   * @brief Compute LogSumExp smooth minimum
   *
   * @param distances Vector of distances
   * @return Smooth minimum using LogSumExp
   */
  double logsumexp_min(const Vec &distances) const;

  /**
   * @brief Compute Voronoi weights for a grid point
   *
   * @param point Grid point coordinates
   * @param atom_positions Atomic positions in Bohr
   * @param atomic_numbers Atomic numbers for VDW radius lookup
   * @return Vec Vector of weights for each atom
   */
  Vec compute_voronoi_weights(const Vec3 &point, const Mat3N &atom_positions,
                              const Eigen::VectorXi &atomic_numbers) const;

  occ::gto::AOBasis m_basis;
  occ::dft::MolecularGrid m_grid;
  std::vector<occ::dft::AtomGrid> m_atom_grids;
  Mat m_density_matrix;
  Vec m_voronoi_charges;
  Vec m_atom_volumes;
  int m_charge{0};
  double m_temperature{0.1};
  bool m_use_vdw_radii{false};
};

/**
 * @brief Calculate Voronoi charges for a molecule
 *
 * Convenience function to compute Voronoi charges in a single call.
 *
 * @param basis Basis set for the molecule
 * @param mo Molecular orbitals containing the density matrix
 * @param charge Total molecular charge
 * @param temperature Temperature parameter for LogSumExp smoothing
 * @param use_vdw_radii Whether to scale distances by VDW radii
 * @return Vec Vector of Voronoi charges for each atom
 */
Vec calculate_voronoi_charges(
    const occ::gto::AOBasis &basis, const occ::qm::MolecularOrbitals &mo,
    int charge = 0, double temperature = 0.1, bool use_vdw_radii = false,
    const occ::io::GridSettings &grid_settings =
        occ::io::GridSettings::from_grid_quality(GridQuality::Coarse));

} // namespace occ::dft