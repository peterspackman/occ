#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cstring>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/units.h>
#include <occ/numint/lebedev.h>
#include <occ/solvent/smd.h>
#include <occ/solvent/surface.h>

namespace occ::solvent::surface {

Mat3 principal_axes(const Mat3N &positions) {
  if (positions.cols() == 1)
    return Mat3::Identity();
  Eigen::JacobiSVD<Mat> svd(positions, Eigen::ComputeThinU);
  Mat3 result = svd.matrixU();
  Vec proportions = svd.singularValues();
  if (proportions.rows() < 3) {
    result.col(2) = result.col(0).cross(result.col(1));
  }
  return result;
}

Surface solvent_surface(const Vec &radii, const IVec &atomic_numbers,
                        const Mat3N &positions, double solvent_radius_angs,
                        bool axis_aligned, double smoothing_width_bohr) {
  const size_t N = atomic_numbers.rows();
  const double solvent_radius =
      std::min(solvent_radius_angs, 0.001) * occ::units::ANGSTROM_TO_BOHR;
  Surface surface;
  auto grid = occ::dft::grid::lebedev(146);
  const int npts = grid.rows();
  Mat tmp_vertices(3, npts * N);
  Vec tmp_areas(npts * N);
  IVec tmp_atom_index(npts * N);
  // For analytical gradients the cavity must be rigidly attached to atoms; the
  // principal-axes rotation introduces a global ∂axes/∂R chain that breaks
  // the rigid-attachment assumption.
  Vec3 centroid = axis_aligned ? Vec3(positions.rowwise().mean())
                               : Vec3(Vec3::Zero());
  Mat3N centered = positions.colwise() - centroid;
  Mat3 axes = axis_aligned ? principal_axes(centered) : Mat3::Identity();
  centered = axes.transpose() * centered;

  Vec ri = radii.array();
  const bool use_smooth = smoothing_width_bohr > 0.0;

  // Pre-shift positions at radius (r_i + solvent_radius) from atom centre.
  // The legacy (boolean) path masks against these and then shifts inward by
  // `solvent_radius` for kept points. The smooth path instead pre-shifts
  // here (tmp_vertices end up at post-shift radius r_i) so the cosmo
  // gradient can reconstruct identical weights from `surface.vertices`
  // alone.
  for (size_t i = 0; i < N; i++) {
    const double rs = ri(i);
    const double r = use_smooth ? rs : (rs + solvent_radius);
    const double surface_area = 4 * M_PI * rs * rs;
    tmp_areas.segment(i * npts, npts).array() =
        grid.col(3).array() * surface_area;
    auto vblock = tmp_vertices.block(0, i * npts, 3, npts);
    vblock.array() = grid.block(0, 0, npts, 3).transpose() * r;
    vblock.colwise() += centered.col(i);
    tmp_atom_index.segment(i * npts, npts).array() = i;
  }

  const size_t n_total = N * npts;
  Vec weights = Vec::Ones(n_total);

  for (size_t i = 0; i < N; i++) {
    const Vec3 q = centered.col(i);
    // Boolean path: cavity point j (pre-shift) is masked by atom i if it
    // lies within (r_i + solvent_radius). Smooth path: post-shift point j
    // is weighted by smoothstep about r_i — both express "point inside
    // atom i's effective sphere", just on different position conventions.
    const double t = use_smooth ? ri(i) : (ri(i) + solvent_radius);
    for (size_t j = 0; j < n_total; j++) {
      if (tmp_atom_index(j) == static_cast<int>(i))
        continue;
      if (weights(j) < 1e-15)
        continue;  // already fully masked, no need to keep multiplying
      const double d = (q - tmp_vertices.col(j)).norm();
      double w;
      if (use_smooth) {
        w = 0.5 * (1.0 + std::erf((d - t) / smoothing_width_bohr));
      } else {
        w = (d < t) ? 0.0 : 1.0;
      }
      weights(j) *= w;
    }
  }

  // Keep points above a tiny weight threshold. Boolean mode is exactly
  // binary so 0.5 picks all "kept" points and reproduces the legacy
  // behaviour bit-for-bit; smooth mode keeps anything still contributing.
  const double keep_threshold = use_smooth ? 1e-3 : 0.5;
  size_t num_kept = 0;
  for (size_t j = 0; j < n_total; j++)
    if (weights(j) > keep_threshold) num_kept++;

  Mat3N remaining_points(3, num_kept);
  Vec remaining_areas(num_kept);
  IVec remaining_atom_index(num_kept);
  size_t k = 0;
  for (size_t j = 0; j < n_total; j++) {
    if (weights(j) <= keep_threshold) continue;
    const size_t atom_idx = tmp_atom_index(j);
    Vec3 v = tmp_vertices.col(j);
    if (!use_smooth) {
      // Legacy convention: kept boolean points get shifted inward by
      // solvent_radius so their final radius is r_atom (not r_atom + probe).
      Vec3 shift = v - centered.col(atom_idx);
      shift.normalize();
      shift *= solvent_radius;
      v -= shift;
    }
    remaining_points.col(k) = v;
    remaining_areas(k) = tmp_areas(j) * weights(j);
    remaining_atom_index(k) = atom_idx;
    k++;
  }
  surface.areas = remaining_areas;
  surface.atom_index = remaining_atom_index;
  surface.vertices = (axes * remaining_points).colwise() + centroid;
  return surface;
}

IVec nearest_atom_index(const Mat3N &atom_positions,
                        const Mat3N &element_centers) {
  IVec result(element_centers.cols());
  Vec3 c, atom;
  for (int i = 0; i < element_centers.cols(); i++) {
    Eigen::Index idx{0};
    double dist = (atom_positions.colwise() - element_centers.col(i))
                      .colwise()
                      .squaredNorm()
                      .minCoeff(&idx);
    result(i) = idx;
  }
  return result;
}

} // namespace occ::solvent::surface
