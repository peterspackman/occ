#pragma once
#include <memory>
#include <occ/dft/grid_types.h>
#include <occ/dft/molecular_grid_points.h>
#include <occ/qm/shell.h>

namespace occ::dft {

using occ::qm::AOBasis;

/**
 * @brief Class for molecular integration grids in DFT
 *
 * This class manages the creation and manipulation of molecular integration
 * grids for density functional theory calculations. It supports various grid
 * generation methods and can work with either basis set information or
 * user-provided data.
 */
class MolecularGrid {
public:
  /**
   * @brief Constructor from AOBasis
   *
   * Creates a molecular grid based on basis set information.
   *
   * @param basis Basis set information
   * @param settings Grid settings
   * @param method Grid generation method
   */
  MolecularGrid(const AOBasis &basis,
                const GridSettings &settings = GridSettings(),
                RadialGridMethod method = RadialGridMethod::LMG);

  /**
   * @brief Constructor from custom positions and atomic numbers
   *
   * Creates a molecular grid based on user-provided atomic positions and
   * numbers.
   *
   * @param positions Atomic positions (3 x N matrix)
   * @param atomic_numbers Atomic numbers
   * @param settings Grid settings
   * @param method Grid generation method
   */
  MolecularGrid(const Mat3N &positions, const IVec &atomic_numbers,
                const GridSettings &settings = GridSettings(),
                RadialGridMethod method = RadialGridMethod::LMG);

  /**
   * @brief Get the number of atoms
   *
   * @return size_t Number of atoms
   */
  size_t n_atoms() const;

  /**
   * @brief Generate a partitioned atom grid for a specific atom
   *
   * @param atom_idx Index of the atom
   * @return AtomGrid The partitioned atom grid
   */
  AtomGrid get_partitioned_atom_grid(size_t atom_idx) const;

  /**
   * @brief Generate an atom grid for a specific atomic number
   *
   * @param atomic_number Atomic number
   * @return AtomGrid The generated atom grid
   */
  AtomGrid get_atom_grid(size_t atom_idx) const;

  /**
   * @brief Get the grid settings
   *
   * @return const GridSettings& The grid settings
   */
  const GridSettings &settings() const;

  /**
   * @brief Generate and initialize the molecular grid points
   *
   * This method generates the actual grid points for the molecular calculation.
   * It is called automatically when needed, but can also be called explicitly.
   */
  void populate_molecular_grid_points();

  /**
   * @brief Get the molecular grid points
   *
   * @return const MolecularGridPoints& The molecular grid points
   */
  const MolecularGridPoints &get_molecular_grid_points() const;


  void set_atomic_radii(const Vec &);
  inline const Vec &atomic_radii() const { return m_atomic_radii; }

private:
  /**
   * @brief Validate and adjust grid settings
   */
  void ensure_settings();

  /**
   * @brief Initialize from basis set information
   *
   * @param basis Basis set information
   */
  void initialize_from_basis(const AOBasis &basis);

  /**
   * @brief Initialize from atomic positions and numbers
   *
   * @param positions Atomic positions
   * @param atomic_numbers Atomic numbers
   */
  void initialize_from_atoms(const Mat3N &positions,
                             const IVec &atomic_numbers);

  void initialize_default_radii();


  IVec m_atomic_numbers;                     ///< Atomic numbers
  Mat3N m_positions;                         ///< Atomic positions
  Mat m_dists;                               ///< Interatomic distances
  Vec m_atomic_radii;                        ///< atomic radii
  bool m_use_custom_radii = false;           ///< Whether to use custom radii
  std::vector<AtomGrid> m_unique_atom_grids; ///< Unique atom grids
  GridSettings m_settings;                   ///< Grid settings
  RadialGridMethod m_radial_method = RadialGridMethod::LMG; ///< Grid method
  mutable MolecularGridPoints m_grid_points; ///< Molecular grid points
  mutable bool m_grid_initialized{false};    ///< Whether grid is initialized

  // Basis-specific information
  IVec m_l_max;    ///< Maximum angular momentum
  Vec m_alpha_max; ///< Maximum exponents
  Mat m_alpha_min; ///< Minimum exponents
};

} // namespace occ::dft
