#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <optional>
#include <string>
#include <vector>

namespace occ::descriptors {

/**
 * \brief Radial basis for the RINSE descriptor.
 *
 * Both families anchor shell *n* by index through a single scale factor, so
 * raising `n_max` extends the descriptor to higher |G| without disturbing the
 * shells already computed.
 */
enum class RinseRadialBasis {
  /// Volume-uniform shells: edges at `scale * n^(1/3)`, each spanning an equal
  /// volume of reciprocal space, each Gaussian scaled so its q^2-weighted
  /// integral is shell-independent. |G|max = scale * n_max^(1/3).
  SmoothShellsNL,
  /// Linearly spaced shells at `scale * n` of width `scale`, row-normalised to
  /// a
  /// partition of unity. |G|max = scale * (n_max - 1).
  SmoothShellsCW,
};

/**
 * \brief Hyper-parameters for the RINSE descriptor.
 *
 * The defaults give the 8 x 16 = 128 element descriptor: radial orders
 * n = 0..7 and angular levels l = 4, 6, ..., 34.
 */
struct RinseParams {
  /// Number of radial shells; radial orders are n = 0 .. n_max-1.
  int n_max{8};
  /// First angular level. 4 drops the monopole and quadrupole terms.
  int l_min{4};
  /// One past the last angular level, so l runs over l_min, l_min+2, ...,
  /// l_max-2.
  int l_max{36};
  /// Per-shell scale in A^-1. Sets the shell spacing, and with `n_max` the
  /// resolution cutoff (see q_max()).
  double radial_scale{0.35};
  RinseRadialBasis radial_basis{RinseRadialBasis::SmoothShellsNL};
  /// Discard the ADPs in the structure and give every atom this isotropic U, in
  /// A^2. Unset (the default) uses the ADPs as read.
  ///
  /// The descriptor is sensitive to ADPs: dropping them can move it a sizeable
  /// fraction of the way to a different polymorph. Set this when comparing
  /// structures that do not all carry ADPs -- a predicted or DFT-relaxed
  /// structure against an experimental one, say -- with the same value for
  /// every structure compared.
  std::optional<double> fixed_uiso{};
  /// Divide each radial level's power by its monopole (l = 0) power, which is
  /// that shell's spherically-averaged scattering. This is what removes the
  /// resolution-dependent intensity envelope, and unlike a fitted envelope it
  /// is unbothered by systematic absences. Applied before log1p and l2.
  bool monopole_normalisation{true};
  /// Compress the power spectrum with log1p. Off by default; useful mainly when
  /// the monopole term is included.
  bool log1p{false};
  /// Scale the whole descriptor to unit L2 norm.
  bool l2{true};

  /// |G| cutoff in A^-1, implied by `radial_scale`, `n_max` and the basis.
  double q_max() const;
  /// The resolution cutoff sin(theta)/lambda = q_max()/2, in A^-1.
  double sin_theta_over_lambda_max() const;
  /// The angular levels, l_min, l_min+2, ..., l_max-2.
  std::vector<int> l_values() const;
  /// Number of angular levels.
  int num_l_levels() const;
  /// Length of the flattened descriptor, n_max * num_l_levels().
  int size() const;

  /// \throws std::invalid_argument if the parameters are not a usable set.
  void validate() const;
};

/**
 * \brief The reflections the descriptor is built from.
 *
 * Only one member of each Friedel pair is stored. Without anomalous dispersion
 * F(-h) = conj(F(h)) exactly -- the same floating point values, not merely the
 * same to rounding -- so `intensities` already carries the factor of two for
 * the mate, and summing over this list is summing over the whole sphere.
 */
struct ReflectionList {
  /// Miller indices, (3, M)
  IMat3N hkl;
  /// Cartesian reciprocal-space vectors G = h a* + k b* + l c*, (3, M), in
  /// A^-1. No factor of 2*pi, so |G| = 2 sin(theta)/lambda.
  Mat3N q_vectors;
  /// |G|, in A^-1
  Vec q_magnitudes;
  /// 2 |F(hkl)|^2, in electrons^2 -- see the note above on the factor of two.
  Vec intensities;

  inline Eigen::Index size() const { return hkl.cols(); }
};

/**
 * \brief Enumerate the resolution sphere and compute reflection intensities.
 *
 * Structure factors come from direct summation over the unit cell with
 * Waasmaier-Kirfel (1995) X-ray form factors -- hydrogen being the bonded
 * Stewart-Davidson-Simpson form factor, see xray_form_factors.h -- and
 * Debye-Waller factors from the structure's ADPs. Anomalous dispersion is not
 * modelled.
 *
 * \throws std::runtime_error if an element has no tabulated form factor.
 */
ReflectionList rinse_reflections(const crystal::Crystal &crystal,
                                 const RinseParams &params = {});

/**
 * \brief Evaluate the radial basis functions.
 * \param q (M,) reciprocal-space magnitudes in A^-1
 * \returns (M, n_max) with element (i, n) = R_n(q_i)
 */
Mat rinse_radial_basis(Eigen::Ref<const Vec> q, const RinseParams &params);

/**
 * \brief RINSE -- Reciprocal-space INvariant Spectral Embedding.
 *
 * A rotationally invariant descriptor of a crystal, built by projecting the
 * intensity-weighted reciprocal lattice onto a radial and angular basis: SOAP,
 * but in reciprocal space. For reflections G within the resolution cutoff,
 *
 * \f[ A_{nlm} = \sum_G I(G) R_n(|G|) Y_{lm}(\hat{G}) \f]
 *
 * and the invariant power spectrum is \f$ p_{nl} = \sum_m A_{nlm}^2 \f$.
 *
 * The intensity field is centrosymmetric absent anomalous dispersion, so odd l
 * cancel exactly and only even l are computed.
 *
 * Constructing a `Rinse` precomputes the shell layout and the harmonic
 * recurrences, so reuse one instance across a batch of structures. It holds no
 * mutable state, so one instance can serve several threads at once.
 *
 * This is a port of rinse-descriptor (Thomas Fellowes,
 * https://github.com/DuMOCC-Group/rinse-descriptor), the reference
 * implementation, and reproduces its descriptors and hashes (tested against
 * version 2.0.0).
 */
class Rinse {
public:
  explicit Rinse(const RinseParams &params = {});

  inline const RinseParams &parameters() const { return m_params; }

  /// The power spectrum for a set of reflections, (n_max, num_l_levels).
  Mat power_spectrum(const ReflectionList &reflections) const;

  /// The power spectrum for a crystal, (n_max, num_l_levels).
  Mat operator()(const crystal::Crystal &crystal) const;

  /// The descriptor as a flat vector of length `size()`, row-major in (n, l):
  /// element n * num_l_levels + k is radial order n and angular level
  /// `l_values()[k]`.
  Vec compute(const crystal::Crystal &crystal) const;

  /// Flatten a power spectrum matrix into the descriptor vector.
  static Vec flatten(Eigen::Ref<const Mat> power_spectrum);

private:
  RinseParams m_params;
  std::vector<int> m_l_values;
  int m_l_max_degree{0};
  int m_num_coefficients{0};
  /// First column of the (2l+1)-wide block for each level
  std::vector<int> m_level_offsets;
  /// Same, indexed by degree instead, or -1 for a degree that is not a level
  std::vector<int> m_degree_columns;
  /// Coefficients of the normalised Legendre recurrence, row l column m
  Mat m_legendre_a;
  Mat m_legendre_b;
  /// Ratio taking the normalised sector harmonic from order m-1 to order m
  Vec m_legendre_sector;
};

/// Number of proquint words produced by rinse_hash() unless asked otherwise.
inline constexpr int default_hash_words = 1;

/**
 * \brief A locality-sensitive hash of a descriptor, as pronounceable proquints.
 *
 * PCA SimHash: centre and project the descriptor onto the leading principal
 * components of a model fitted to the CSD, take the sign of each coefficient as
 * a bit, and encode each 16 bits as a five-character CVCVC proquint word.
 * Similar structures give the same or a nearby string, e.g. "lusab-babad".
 *
 * \param descriptor a flat 128-element descriptor, as returned by
 *        Rinse::compute() with the default parameters
 * \param num_words words in the output; each carries 16 bits, at most 8 words
 *
 * \throws std::invalid_argument if the descriptor is the wrong length or more
 *         words are asked for than the bundled model has components.
 */
std::string rinse_hash(Eigen::Ref<const Vec> descriptor,
                       int num_words = default_hash_words);

/// Decode a proquint hash back to its bits, most significant first.
std::vector<bool> rinse_hash_to_bits(std::string_view hash);

} // namespace occ::descriptors
