#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/dma.h>
#include <occ/dma/mult.h>
#include <occ/qm/wavefunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace occ::dma {

struct ProductPrimitive {
  Vec3 center;
  double exponent;
  double coefficient;
  int l{0};
};

/**
 * @brief Handles analytical calculation of multipole moments using
 * Gauss-Hermite quadrature
 */
class AnalyticalIntegrator {
public:
  AnalyticalIntegrator(const DMASettings &settings);

  /**
   * @brief Calculate multipole contribution from a primitive pair
   */
  void calculate_primitive_contribution(const qm::Shell &shell_i,
                                        const qm::Shell &shell_j, int i_prim,
                                        int j_prim, double fac,
                                        const Mat &d_block, const Vec3 &P,
                                        Mult &qt) const;

private:
  const DMASettings &m_settings;
  mutable Eigen::Tensor<double, 3> m_gx, m_gy, m_gz;
};

/**
 * @brief Handles numerical integration on grids
 */
class GridIntegrator {
public:
  GridIntegrator(const DMASettings &settings);

  /**
   * @brief Add primitive contribution to grid density
   */
  void add_primitive_to_grid(const qm::Shell &shell_i, const qm::Shell &shell_j,
                             int i_prim, int j_prim, double fac,
                             const Mat &d_block, const Vec3 &P,
                             const Mat3N &grid_points, Vec &rho,
                             double etol) const;

  void add_primitive_contributions(const std::vector<ProductPrimitive> &pps,
                                   const Mat &d_block, const Mat3N &grid_points,
                                   Vec &rho) const;

  /**
   * @brief Process grid density to extract multipoles
   */
  void process_grid_density(
      const Vec &rho, const Mat3N &grid_points, const Vec &grid_weights,
      const std::vector<std::pair<size_t, size_t>> &atom_blocks,
      const DMASites &sites, std::vector<Mult> &site_multipoles) const;

private:
  const DMASettings &m_settings;
};

/**
 * @brief Main calculator for multipole moments
 */
class MultipoleCalculator {
public:
  MultipoleCalculator(const qm::AOBasis &basis, const qm::MolecularOrbitals &mo,
                      const DMASites &sites, const DMASettings &settings);

  /**
   * @brief Calculate all multipole moments
   */
  std::vector<Mult> calculate();

private:
  void setup_normalized_density_matrix();
  void process_nuclear_contributions(std::vector<Mult> &site_multipoles);
  void process_electronic_contributions(std::vector<Mult> &site_multipoles);

  const qm::AOBasis &m_basis;
  const qm::MolecularOrbitals &m_mo;
  const DMASites &m_sites;
  const DMASettings &m_settings;

  Mat m_normalized_density;
  AnalyticalIntegrator m_analytical;
  GridIntegrator m_grid;

  // Caching/optimization members
  double m_tolerance;
  bool m_use_quadrature;
  Vec m_grid_density;
};

} // namespace occ::dma
