#pragma once
#include <string>
#include <Eigen/Geometry>
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <occ/crystal/unitcell.h>
#include <vector>

namespace occ::crystal {

class Crystal;

/**
 * \brief Results from analyzing a crystal surface cut
 *
 * Contains information about molecules and dimer counts in different regions
 * relative to a surface cut, including those above, below, in the slab, and in
 * the bulk.
 */
struct SurfaceCutResult {
  using DimerCounts = std::vector<std::vector<int>>;

  /**
   * \brief Constructor initializing the result from crystal dimers
   * \param dimers The crystal dimers to analyze
   */
  SurfaceCutResult(const CrystalDimers &);
  std::vector<Molecule> molecules;
  std::string exyz;
  DimerCounts above;
  DimerCounts below;
  DimerCounts slab;
  DimerCounts bulk;
  double depth_scale{1.0};
  Mat3 basis;
  double cut_offset{0.0};

  /**
   * \brief Calculate total dimer count above the surface
   * \param dimers Reference crystal dimers
   * \return Total number of dimers above the surface
   */
  /// Sum a named per-dimer interaction component over the dimers cut by the
  /// surface. Defaults to the total interaction energy; any other channel
  /// carried on the dimer (a solvation descriptor, say) sums the same way.
  double total_above(const CrystalDimers &,
                     const std::string &key = "Total") const;
  double total_below(const CrystalDimers &,
                     const std::string &key = "Total") const;
  double total_slab(const CrystalDimers &,
                    const std::string &key = "Total") const;
  double total_bulk(const CrystalDimers &,
                    const std::string &key = "Total") const;
  std::vector<std::vector<size_t>>
  unique_counts_above(const CrystalDimers &) const;
};

/**
 * \brief Represents and analyzes a crystal surface defined by Miller indices
 */
class Surface {
public:
  /**
   * \brief Construct a surface from Miller indices and crystal
   * \param hkl Miller indices defining the surface
   * \param crystal Reference crystal structure
   */
  Surface(const HKL &, const Crystal &);

  double depth() const;
  double d() const;
  void print() const;

  /**
   * \brief Get the surface normal vector
   * \return Vector perpendicular to the surface
   */
  Vec3 normal_vector() const;
  inline const auto &hkl() const { return m_hkl; }
  inline const auto &depth_vector() const { return m_depth_vector; }
  inline const auto &a_vector() const { return m_a_vector; };
  inline const auto &b_vector() const { return m_b_vector; };
  inline double area() const { return m_a_vector.cross(m_b_vector).norm(); }
  Vec3 dipole() const;

  Mat3 basis_matrix(double depth_scale = 1.0) const;

  // Pack unit cell molecules below or above the surface with depth
  // negative depth means below, positive depth means above (as determined by
  // surface normal), depth is in fractions of the depth of the surface i.e.
  // interplanar spacing
  std::vector<Molecule>
  find_molecule_cell_translations(const std::vector<Molecule> &unit_cell_mols,
                                  double depth, double cut_offset = 0.0);

  SurfaceCutResult count_crystal_dimers_cut_by_surface(const CrystalDimers &,
                                                       double cut_offset = 0.0);

  std::vector<double> possible_cuts(Eigen::Ref<const Mat3N> unique_positions,
                                    double epsilon = 1e-6) const;

  static bool check_systematic_absence(const Crystal &, const HKL &);
  static bool faces_are_equivalent(const Crystal &, const HKL &, const HKL &);

private:
  HKL m_hkl;
  double m_depth{0.0};
  UnitCell m_crystal_unit_cell;
  Vec3 m_a_vector;
  Vec3 m_b_vector;
  Vec3 m_depth_vector;
  Vec3 m_dipole{0.0, 0.0, 0.0};
  double m_angle{0.0};
};

/**
 * \brief Parameters for crystal surface generation
 */
struct CrystalSurfaceGenerationParameters {
  double d_min{0.1};  ///< Minimum d-spacing
  double d_max{1.0};  ///< Maximum d-spacing
  bool unique{true};  ///< Generate only symmetry-unique surfaces
  bool reduced{true}; ///< Use reduced Miller indices
  bool systematic_absences_allowed{true}; ///< Allow systematic absences
};

std::vector<Surface>
generate_surfaces(const Crystal &c,
                  const CrystalSurfaceGenerationParameters & = {});

/// Whether the crystal's point group maps `hkl` onto `-h-k-l`, i.e. whether
/// the two faces belong to the same form.
///
/// True for the 11 centrosymmetric point groups. False for the other 21,
/// where (hkl) and (-h-k-l) are distinct forms — this is hemimorphism, and
/// the two ends of a polar axis can carry different terminations and
/// different surface energies.
///
/// The pair is nonetheless *always* degenerate in d, because 1/d² = hᵀG*h is
/// a quadratic form in the indices and so is invariant under h → -h,
/// whatever the metric. Ordering faces by d therefore cannot separate a
/// Friedel pair: d has the symmetry of the Laue class, while equivalence of
/// faces has the symmetry of the point group, and for the non-centrosymmetric
/// groups the Laue class has index two over it. Anything that truncates a
/// d-ordered list has to cut on whole Laue orbits or it splits such a pair
/// arbitrarily, which is what `laue_orbit_partners` is for.
bool friedel_mate_is_equivalent(const Crystal &c, const HKL &hkl);

/// Positions in `surfaces` that share a Laue-class orbit with `surfaces[i]`,
/// including `i` itself.
///
/// A single form when the point group is centrosymmetric, a Friedel pair of
/// forms otherwise. Truncating a d-ordered surface list on these groups keeps
/// the cut crystallographically meaningful.
std::vector<size_t> laue_orbit_partners(const Crystal &c,
                                        const std::vector<Surface> &surfaces,
                                        size_t i);

} // namespace occ::crystal
