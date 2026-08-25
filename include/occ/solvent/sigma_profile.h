#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/surface.h>

/// Screening-charge-density (σ) machinery shared by the COSMO-SAC and
/// COSMO-RS solvation models.
///
/// Units follow the published parameter tables and are converted at the
/// boundary: σ in e/Å², segment areas in Å², distances in Å (positions are
/// stored in Bohr to match `surface::Surface`).
namespace occ::solvent::sigma {

/// Uniform grid over σ (e/Å²). The conventional choice is 51 bins over
/// ±0.025 e/Å².
struct Grid {
  int n{51};
  double lo{-0.025};
  double hi{0.025};

  /// Bin centres, low to high. Length `n`.
  Vec centers() const;
  double spacing() const;
  bool operator==(const Grid &other) const;
  bool operator!=(const Grid &other) const { return !(*this == other); }
};

/// H-bond character of a segment's parent atom (COSMO-SAC 2010). `OH` is a
/// hydrogen bound to O, N or F; `OT` is O, N or F itself; `None` (NHB) is
/// everything else.
enum class HBondClass : int { None = 0, OH = 1, OT = 2 };
inline constexpr int num_hbond_classes = 3;

/// Per-segment COSMO data.
struct Segments {
  Mat3N positions;    ///< Bohr
  Vec areas;          ///< Å²
  Vec sigma;          ///< e/Å², raw q_i / a_i
  Vec sigma_averaged; ///< e/Å², after `average_sigma`
  IVec atom_index;
  IVec hbond_class; ///< `HBondClass` values; all `None` until classified

  Eigen::Index size() const { return areas.size(); }
  double total_area() const { return areas.sum(); }
  /// Σ a_i σ_i. For a conductor cavity this approaches −q_solute; the
  /// deviation measures the cavity discretisation error.
  double total_charge() const;
  double total_charge_averaged() const;
};

/// Build segments from a COSMO cavity and the apparent surface charges.
///
/// `charges` are segment charges in e, not densities — the COSMO A matrix
/// carries units of inverse length, so `σ = A⁻¹(−f φ)` is a charge.
Segments segments_from_cavity(const surface::Surface &cavity,
                              const Vec &charges);

/// Klamt segment averaging onto the effective contact scale:
///
///     σ̄_i = Σ_j σ_j w_ij / Σ_j w_ij
///     w_ij = (r_j² r_av²)/(r_j² + r_av²) · exp[ −d_ij²/(r_j² + r_av²) ]
///
/// with `r_j = √(a_j/π)` the radius of an equal-area disc. Fills
/// `sigma_averaged`. The Gaussian is truncated where it falls below ~1e-12.
void average_sigma(Segments &segments, double r_av_angs = 0.5);

/// Assign `hbond_class` from atomic numbers and geometry. Connectivity is
/// perceived with the covalent-radius criterion the crystal module uses,
/// `d < r_cov(a) + r_cov(b) + core::covalent_bond_tolerance`.
void classify_hbond_segments(Segments &segments, const IVec &atomic_numbers,
                             const Mat3N &atom_positions_bohr);

/// Binned σ-profile holding area per bin, in Å² (not normalised — the total
/// area sets the segment count `A/a_eff`).
///
/// `values` is `n × num_classes`: one column for an H-bond-agnostic model
/// (COSMO-SAC 2002, Klamt), three for the COSMO-SAC 2010 split.
struct Profile {
  Grid grid;
  Mat values;

  int num_classes() const { return static_cast<int>(values.cols()); }
  double total_area() const { return values.sum(); }
  /// Column-summed profile, length `n`.
  Vec total() const;
  /// `values / total_area()`, so the entries sum to 1.
  Mat normalized() const;
};

/// Bin segments onto `grid` using `sigma_averaged`, spreading each segment's
/// area linearly between its two neighbouring bin centres. Segments outside
/// the grid are clamped onto the end bins; their total area is reported via
/// `out_of_range_area` when non-null.
Profile bin_segments(const Segments &segments, const Grid &grid,
                     bool resolve_hbond_classes = false,
                     double *out_of_range_area = nullptr);

/// Mixture profile `Σ_k x_k A_k p_k(σ)`. All components must share a grid
/// and class count.
Profile mix_profiles(const std::vector<Profile> &components,
                     const Vec &mole_fractions);

/// `Σ_{bin,class} values(bin, class) · field(bin, class)`.
double contract(const Profile &profile, const Mat &field);

/// Segment-resolved contraction `a_i · field(σ̄_i, class_i)`, with `field`
/// linearly interpolated in σ. Length `segments.size()`.
Vec contract_segments(const Segments &segments, const Grid &grid,
                      const Mat &field);

} // namespace occ::solvent::sigma
