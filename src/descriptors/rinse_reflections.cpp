#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/symmetryoperation.h>
#include <occ/crystal/xray_form_factors.h>
#include <occ/descriptors/rinse.h>
#include <stdexcept>
#include <tuple>

namespace occ::descriptors {

using crystal::Crystal;
using crystal::HKL;
using crystal::SymmetryOperation;
using crystal::XrayFormFactor;

namespace {

constexpr double two_pi = 2.0 * occ::units::PI;
constexpr double two_pi_sq = 2.0 * occ::units::PI * occ::units::PI;

/// Reflections in blocks of this many. Small enough that the reflection-indexed
/// scratch stays in L1, large enough to amortise the per-block setup.
constexpr Eigen::Index reflection_block = 512;

/// Reflection-indexed scratch. Row shaped so it lines up with the rows of the
/// (3, M) Miller index matrix without a transpose in the inner loop.
using RowArray = Eigen::Array<double, 1, Eigen::Dynamic>;

/**
 * The unit cell atoms, reordered so that each element's atoms are contiguous.
 * The form factor then only needs evaluating once per element per reflection
 * rather than once per atom.
 */
struct Scatterers {
  Mat3N frac;
  Vec occupation;
  /// -2 pi^2 times u_star, packed (u11, u22, u33, 2 u12, 2 u13, 2 u23) so a
  /// Debye-Waller exponent is a plain dot product with (h^2, k^2, l^2, hk, hl,
  /// kl)
  Mat6N debye_waller;
  bool any_debye_waller{false};
  std::vector<XrayFormFactor> form_factors;
  /// Row range in `frac` belonging to element e, size form_factors.size() + 1
  std::vector<Eigen::Index> element_offsets;
};

/// The reciprocal metric tensor, contracted with (h^2, k^2, l^2, hk, hl, kl).
Vec6 packed_metric(const Mat3 &metric) {
  Vec6 result;
  result << metric(0, 0), metric(1, 1), metric(2, 2), 2.0 * metric(0, 1),
      2.0 * metric(0, 2), 2.0 * metric(1, 2);
  return result;
}

/**
 * u_star for every atom in the cell, in the packed convention above.
 *
 * ADPs are stored in the u_cif convention, U^ij referred to the reciprocal
 * axes, so u_star = U^ij a*_i a*_j. Each unit cell atom is a symmetry image of
 * an asymmetric unit site and its tensor rotates with it: u_star' = R u_star
 * R^T.
 */
Mat6N unit_cell_debye_waller(const Crystal &crystal,
                             const std::optional<double> &fixed_uiso) {
  const auto &atoms = crystal.unit_cell_atoms();
  const Eigen::Index num_atoms = atoms.size();
  const Mat3 metric = crystal.unit_cell().reciprocal_metric_tensor();

  if (fixed_uiso) {
    // An isotropic U is u_star = U * (reciprocal metric): the Debye-Waller
    // exponent collapses to -8 pi^2 U (sin(theta)/lambda)^2.
    const Vec6 u = -two_pi_sq * (*fixed_uiso) * packed_metric(metric);
    return u.replicate(1, num_atoms);
  }

  const auto &asym = crystal.asymmetric_unit();
  Mat6N result = Mat6N::Zero(6, num_atoms);
  if (asym.adps.cols() != asym.positions.cols() || asym.adps.isZero(0.0))
    return result;

  const Vec3 astar = crystal.unit_cell().reciprocal().colwise().norm();
  // The (u11, u22, u33, u12, u13, u23) packing, scaled by a*_i a*_j.
  const Vec6 scale =
      (Vec6() << astar(0) * astar(0), astar(1) * astar(1), astar(2) * astar(2),
       astar(0) * astar(1), astar(0) * astar(2), astar(1) * astar(2))
          .finished();

  for (Eigen::Index i = 0; i < num_atoms; i++) {
    const Vec6 u_star_asym =
        asym.adps.col(atoms.asym_idx(i)).cwiseProduct(scale);
    const Vec6 u = SymmetryOperation(atoms.symop(i)).rotate_adp(u_star_asym);
    result.col(i) << u(0), u(1), u(2), 2.0 * u(3), 2.0 * u(4), 2.0 * u(5);
  }
  result *= -two_pi_sq;
  return result;
}

Scatterers build_scatterers(const Crystal &crystal,
                            const std::optional<double> &fixed_uiso) {
  const auto &atoms = crystal.unit_cell_atoms();
  const Eigen::Index num_atoms = atoms.size();
  const Mat6N debye_waller = unit_cell_debye_waller(crystal, fixed_uiso);

  std::vector<Eigen::Index> order(num_atoms);
  std::iota(order.begin(), order.end(), Eigen::Index{0});
  std::stable_sort(order.begin(), order.end(),
                   [&atoms](Eigen::Index a, Eigen::Index b) {
                     return atoms.atomic_numbers(a) < atoms.atomic_numbers(b);
                   });

  Scatterers result;
  result.frac.resize(3, num_atoms);
  result.occupation.resize(num_atoms);
  result.debye_waller.resize(6, num_atoms);
  result.element_offsets.push_back(0);

  int current_z = -1;
  for (Eigen::Index i = 0; i < num_atoms; i++) {
    const Eigen::Index j = order[i];
    result.frac.col(i) = atoms.frac_pos.col(j);
    result.occupation(i) = atoms.occupation(j);
    result.debye_waller.col(i) = debye_waller.col(j);

    const int z = atoms.atomic_numbers(j);
    if (z != current_z) {
      const auto ff = crystal::xray_form_factor(z);
      if (!ff)
        throw std::runtime_error(fmt::format(
            "No Waasmaier-Kirfel X-ray form factor for element {} (Z={})",
            core::Element(z).symbol(), z));
      result.form_factors.push_back(*ff);
      if (i > 0)
        result.element_offsets.push_back(i);
      current_z = z;
    }
  }
  result.element_offsets.push_back(num_atoms);
  result.any_debye_waller = !result.debye_waller.isZero(0.0);
  return result;
}

/**
 * exp(2 pi i n r) for every atom, every axis and every Miller index in range.
 *
 * The phase factor of a reflection is separable,
 * exp(2 pi i h.r) = exp(2 pi i h x) exp(2 pi i k y) exp(2 pi i l z), so
 * tabulating the three one-dimensional factors turns the inner loop's sine and
 * cosine -- scalar library calls, and by far the most expensive thing in it --
 * into two complex multiplications. There are only tens of table entries per
 * atom against hundreds of reflections, so the tables pay for themselves
 * several times over.
 */
class PhaseTables {
public:
  PhaseTables(const Mat3N &fractional, const HKL &extent)
      : m_stride{extent.h + 1, extent.k + 1, extent.l + 1} {
    const Eigen::Index num_atoms = fractional.cols();
    for (int axis = 0; axis < 3; axis++) {
      auto &table = m_table[axis];
      const int stride = m_stride[axis];
      table.resize(num_atoms * stride);
      for (Eigen::Index j = 0; j < num_atoms; j++) {
        std::complex<double> *entries = table.data() + j * stride;
        entries[0] = {1.0, 0.0};
        if (stride > 1)
          entries[1] = std::polar(1.0, two_pi * fractional(axis, j));
        // Built by halving rather than by repeated multiplication so the error
        // grows with the logarithm of the index, not the index.
        for (int n = 2; n < stride; n++)
          entries[n] = entries[n / 2] * entries[n - n / 2];
      }
    }
  }

  /// The factor for atom `atom` on `axis`; negative indices are the conjugate.
  inline std::complex<double> at(int axis, Eigen::Index atom, int n) const {
    const std::complex<double> value =
        m_table[axis][atom * m_stride[axis] + std::abs(n)];
    return n < 0 ? std::conj(value) : value;
  }

private:
  std::array<int, 3> m_stride;
  std::array<std::vector<std::complex<double>>, 3> m_table;
};

/// Scratch arrays for one block of reflections.
struct BlockWorkspace {
  explicit BlockWorkspace(Eigen::Index n)
      : hh(6, n), stol_sq(n), form_factor(n), debye_waller(n), amplitude(n),
        real_part(n), imag_part(n), element_real(n), element_imag(n) {}
  Mat hh;
  RowArray stol_sq;
  RowArray form_factor;
  /// A matrix product needs somewhere to land; without this Eigen allocates a
  /// temporary for it once per atom per block, which is most of the cost.
  Eigen::RowVectorXd debye_waller;
  RowArray amplitude;
  RowArray real_part;
  RowArray imag_part;
  RowArray element_real;
  RowArray element_imag;
};

void structure_factors_block(const Scatterers &scatterers,
                             const PhaseTables &phases, const IMat3N &hkl,
                             const Vec &d_star_sq, Eigen::Index begin,
                             Eigen::Index count, BlockWorkspace &work,
                             Eigen::Ref<Vec> intensities) {
  const auto h = hkl.block(0, begin, 3, count);
  const Mat h_double = h.cast<double>();
  work.hh.row(0) = h_double.row(0).array().square();
  work.hh.row(1) = h_double.row(1).array().square();
  work.hh.row(2) = h_double.row(2).array().square();
  work.hh.row(3) = h_double.row(0).array() * h_double.row(1).array();
  work.hh.row(4) = h_double.row(0).array() * h_double.row(2).array();
  work.hh.row(5) = h_double.row(1).array() * h_double.row(2).array();

  // (sin(theta)/lambda)^2 = |G|^2 / 4
  work.stol_sq = 0.25 * d_star_sq.segment(begin, count).transpose().array();

  work.real_part.setZero();
  work.imag_part.setZero();

  const int num_elements = static_cast<int>(scatterers.form_factors.size());
  for (int e = 0; e < num_elements; e++) {
    const XrayFormFactor &ff = scatterers.form_factors[e];
    work.form_factor.setConstant(ff.c);
    for (int g = 0; g < 5; g++)
      work.form_factor += ff.a[g] * (-ff.b[g] * work.stol_sq).exp();

    work.element_real.setZero();
    work.element_imag.setZero();
    for (Eigen::Index j = scatterers.element_offsets[e];
         j < scatterers.element_offsets[e + 1]; j++) {
      if (scatterers.any_debye_waller) {
        work.debye_waller.noalias() =
            scatterers.debye_waller.col(j).transpose() * work.hh;
        work.amplitude =
            scatterers.occupation(j) * work.debye_waller.array().exp();
      } else {
        work.amplitude.setConstant(scatterers.occupation(j));
      }

      // Reflections arrive in (h, k, l) order with l innermost, so consecutive
      // ones usually share h and k and the first factor can be carried over.
      int last_h = std::numeric_limits<int>::min();
      int last_k = std::numeric_limits<int>::min();
      std::complex<double> in_plane{1.0, 0.0};
      for (Eigen::Index b = 0; b < count; b++) {
        if (h(0, b) != last_h || h(1, b) != last_k) {
          last_h = h(0, b);
          last_k = h(1, b);
          in_plane = phases.at(0, j, last_h) * phases.at(1, j, last_k);
        }
        const std::complex<double> phase = in_plane * phases.at(2, j, h(2, b));
        work.element_real(b) += work.amplitude(b) * phase.real();
        work.element_imag(b) += work.amplitude(b) * phase.imag();
      }
    }
    work.real_part += work.form_factor * work.element_real;
    work.imag_part += work.form_factor * work.element_imag;
  }

  // The Friedel mate contributes an identical intensity: F(-h) is the exact
  // conjugate of F(h), down to the last bit, because negating integer Miller
  // indices conjugates every tabulated phase factor. Counting it here is what
  // lets the whole descriptor be built from half the sphere.
  intensities.segment(begin, count) =
      2.0 *
      (work.real_part.square() + work.imag_part.square()).matrix().transpose();
}

/**
 * Map each reflection to the index of its Laue-group orbit representative.
 *
 * The representative is the lexicographically greatest member of the orbit,
 * which is always in the half sphere the caller enumerated: if v is the
 * greatest then -v is the smallest, so v is lexicographically positive.
 */
std::vector<Eigen::Index>
laue_orbit_representatives(const crystal::SpaceGroup &space_group,
                           const std::vector<IVec3> &indices) {
  const std::vector<Eigen::Matrix3i> rotations = space_group.laue_rotations();

  ankerl::unordered_dense::map<int64_t, Eigen::Index> index_of;
  index_of.reserve(indices.size());
  // Three 21-bit fields, biased so negative indices stay inside their own
  // field. Miller indices anywhere near the million this allows would need a
  // unit cell the size of a virus.
  const auto key = [](const IVec3 &v) {
    constexpr int64_t bias = 1 << 20;
    return ((static_cast<int64_t>(v(0)) + bias) << 42) |
           ((static_cast<int64_t>(v(1)) + bias) << 21) |
           (static_cast<int64_t>(v(2)) + bias);
  };
  for (size_t i = 0; i < indices.size(); i++)
    index_of.emplace(key(indices[i]), static_cast<Eigen::Index>(i));

  const auto lexicographically_greater = [](const IVec3 &a, const IVec3 &b) {
    return std::tie(a(0), a(1), a(2)) > std::tie(b(0), b(1), b(2));
  };

  std::vector<Eigen::Index> result(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    IVec3 best = indices[i];
    for (const auto &rotation : rotations) {
      const IVec3 image = rotation * indices[i];
      if (lexicographically_greater(image, best))
        best = image;
    }
    const auto found = index_of.find(key(best));
    // Rotations preserve |G|, so the whole orbit is inside the resolution
    // sphere and the representative is always in the list.
    result[i] =
        found != index_of.end() ? found->second : static_cast<Eigen::Index>(i);
  }
  return result;
}

} // namespace

ReflectionList rinse_reflections(const Crystal &crystal,
                                 const RinseParams &params) {
  params.validate();

  const auto &unit_cell = crystal.unit_cell();
  const Mat3 &reciprocal = unit_cell.reciprocal();
  const Mat3 metric = unit_cell.reciprocal_metric_tensor();

  // Round-trip the cutoff through d_min the way cctbx does. The two forms are
  // the same number in exact arithmetic but not in floating point, and which
  // one is used decides the fate of a reflection sitting exactly on the edge.
  const double d_min = 1.0 / (2.0 * params.sin_theta_over_lambda_max());
  const double d_star_sq_max = 1.0 / (d_min * d_min);
  const HKL limits = unit_cell.hkl_limits(d_min);

  // Only one member of each Friedel pair: the lexicographically positive one,
  // h > 0, or h = 0 and k > 0, or the (0, 0, l > 0) axis.
  std::vector<IVec3> indices;
  std::vector<double> d_star_sq_values;
  const auto try_add = [&](int h, int k, int l) {
    const IVec3 v(h, k, l);
    const Vec3 real_v = v.cast<double>();
    const double dss = real_v.dot(metric * real_v);
    if (dss > d_star_sq_max)
      return;
    indices.push_back(v);
    d_star_sq_values.push_back(dss);
  };
  for (int k = 1; k <= limits.k; k++)
    for (int l = -limits.l; l <= limits.l; l++)
      try_add(0, k, l);
  for (int l = 1; l <= limits.l; l++)
    try_add(0, 0, l);
  for (int h = 1; h <= limits.h; h++)
    for (int k = -limits.k; k <= limits.k; k++)
      for (int l = -limits.l; l <= limits.l; l++)
        try_add(h, k, l);

  const Eigen::Index num_reflections =
      static_cast<Eigen::Index>(indices.size());
  ReflectionList result;
  result.hkl.resize(3, num_reflections);
  result.intensities.resize(num_reflections);
  Vec d_star_sq(num_reflections);
  for (Eigen::Index i = 0; i < num_reflections; i++) {
    result.hkl.col(i) = indices[i];
    d_star_sq(i) = d_star_sq_values[i];
  }
  result.q_vectors = reciprocal * result.hkl.cast<double>();
  result.q_magnitudes = d_star_sq.cwiseSqrt();

  if (num_reflections == 0)
    return result;

  // Reflections in one orbit of the Laue group share an intensity exactly, so
  // only one of each needs a structure factor. That is a factor of two for a
  // monoclinic cell and up to twenty-four for a cubic one.
  const std::vector<Eigen::Index> orbit_representative =
      laue_orbit_representatives(crystal.space_group(), indices);
  std::vector<Eigen::Index> unique;
  unique.reserve(num_reflections);
  for (Eigen::Index i = 0; i < num_reflections; i++)
    if (orbit_representative[i] == i)
      unique.push_back(i);

  IMat3N unique_hkl(3, unique.size());
  Vec unique_d_star_sq(unique.size());
  for (size_t i = 0; i < unique.size(); i++) {
    unique_hkl.col(i) = result.hkl.col(unique[i]);
    unique_d_star_sq(i) = d_star_sq(unique[i]);
  }

  const Scatterers scatterers = build_scatterers(crystal, params.fixed_uiso);
  const PhaseTables phases(scatterers.frac, limits);
  occ::log::debug("RINSE: {} reflections (half sphere), {} unique under the "
                  "Laue group, |G| <= {:.4f} A^-1, {} scatterers in the cell",
                  num_reflections, unique.size(), params.q_max(),
                  scatterers.frac.cols());

  const Eigen::Index num_unique = static_cast<Eigen::Index>(unique.size());
  Vec unique_intensities(num_unique);
  const Eigen::Index num_blocks =
      (num_unique + reflection_block - 1) / reflection_block;
  occ::parallel::parallel_for(
      size_t{0}, static_cast<size_t>(num_blocks), [&](size_t block) {
        const Eigen::Index begin =
            static_cast<Eigen::Index>(block) * reflection_block;
        const Eigen::Index count =
            std::min(reflection_block, num_unique - begin);
        BlockWorkspace work(count);
        structure_factors_block(scatterers, phases, unique_hkl,
                                unique_d_star_sq, begin, count, work,
                                unique_intensities);
      });

  // Scatter back. `orbit_representative` indexes the half-sphere list, and the
  // representatives were taken from it in order, so a running index recovers
  // where each one landed.
  std::vector<Eigen::Index> position(num_reflections, 0);
  for (Eigen::Index i = 0; i < num_unique; i++)
    position[unique[i]] = i;
  for (Eigen::Index i = 0; i < num_reflections; i++)
    result.intensities(i) =
        unique_intensities(position[orbit_representative[i]]);

  return result;
}

} // namespace occ::descriptors
