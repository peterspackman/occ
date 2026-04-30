#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/parallel.h>
#include <occ/crystal/crystal.h>
#include <occ/isosurface/atom_cell_list.h>
#include <occ/isosurface/common.h>

namespace occ::isosurface {

// Marching-cubes functor for crystal void surfaces.
//
// Sampling is in *fractional* coordinates over [0, 1]^3 (the canonical unit
// cell), endpoint-inclusive so frac=0 and frac=1 land exactly on the cell
// faces. Per-axis sample counts scale with lattice lengths so the cartesian
// voxel edge is approximately the requested separation.
//
// Density is evaluated by summing radial Slater densities of atoms in a
// pre-built slab around the unit cell (extended by m_buffer); the slab is
// large enough that values are effectively periodic across the cell faces,
// so the same fractional point on opposite faces yields the same density.
//
// The mesh is therefore *open* where the void surface crosses a cell face:
// vertex positions on the shared face are determined entirely by the four
// corner values on that face, which are bit-identical between neighbouring
// cells. This gives exact cell-boundary alignment for stitching. Caps for
// closed per-cell volumes can be added as a post-process if needed.
class VoidSurfaceFunctor {
public:
  // Padding (Angstroms) added in each direction when building the slab of
  // periodic-image atoms. Large enough that the summed Slater density is
  // effectively periodic across the cell faces. Volume/cube output must use
  // the same buffer to produce a matching scalar field.
  static constexpr double DEFAULT_BUFFER_ANGSTROM = 8.0;

  VoidSurfaceFunctor(const crystal::Crystal &crystal, float sep,
                     const InterpolatorParams &params = {});

  // Vertices come out of MC in fractional coords; snap any vertex within a
  // tight tolerance of a cell face to the face exactly (rounds away tiny
  // float errors), then transform to cartesian angstroms via the unit cell
  // direct matrix.
  static constexpr float SNAP_TOL = 1.0e-5f;

  inline void remap_vertices(const std::vector<float> &v,
                             std::vector<float> &dest) const {
    dest.resize(v.size());
    const Mat3 &M = m_crystal.unit_cell().direct(); // angstroms
    for (size_t i = 0; i < v.size(); i += 3) {
      Vec3 frac(v[i], v[i + 1], v[i + 2]);
      for (int k = 0; k < 3; k++) {
        if (frac(k) < SNAP_TOL)
          frac(k) = 0.0;
        else if (frac(k) > 1.0 - SNAP_TOL)
          frac(k) = 1.0;
      }
      Vec3 cart = M * frac;
      dest[i] = static_cast<float>(cart(0));
      dest[i + 1] = static_cast<float>(cart(1));
      dest[i + 2] = static_cast<float>(cart(2));
    }
  }

  // Evaluate density at a single fractional point. No clipping: the slab
  // buffer makes the function effectively periodic over [0, 1]^3.
  inline float operator()(const FVec3 &frac) const {
    m_num_calls++;
    Eigen::Vector3f pos = m_direct_bohr * frac;
    float result{0.0f};
    m_cell_list.for_each_close(
        pos, [&](uint32_t gi, float r_sq, const Eigen::Vector3f &) {
          result += m_atom_interpolators[gi].interpolator(r_sq);
        });
    return result;
  }

  inline void batch(Eigen::Ref<const FMat3N> frac_pos,
                    Eigen::Ref<FVec> layer) const {
    occ::parallel::parallel_for(0, int(frac_pos.cols()), [&](int pt) {
      m_num_calls++;
      Eigen::Vector3f p = m_direct_bohr * frac_pos.col(pt);
      float tot = 0.0f;
      m_cell_list.for_each_close(
          p, [&](uint32_t gi, float r_sq, const Eigen::Vector3f &) {
            tot += m_atom_interpolators[gi].interpolator(r_sq);
          });
      layer(pt) = tot;
    });
  }

  // Gradient in cartesian (bohr) at a fractional point. Used outside MC.
  OCC_ALWAYS_INLINE Eigen::Vector3f normal(float fx, float fy, float fz) const {
    Eigen::Vector3f frac(fx, fy, fz);
    Eigen::Vector3f pos = m_direct_bohr * frac;
    Eigen::Vector3f grad(0.0f, 0.0f, 0.0f);
    m_num_calls++;
    m_cell_list.for_each_close(
        pos, [&](uint32_t gi, float r_sq, const Eigen::Vector3f &delta) {
          float grad_rho = m_atom_interpolators[gi].interpolator.gradient(r_sq);
          grad.array() += 2.0f * delta.array() * grad_rho;
        });
    return grad.normalized();
  }

  // MC samples in fractional coords on [0, 1]^3.
  inline const FVec3 &origin() const { return m_origin; }
  inline const FVec3 &side_length() const { return m_side_length; }
  inline Eigen::Vector3i cubes_per_side() const { return m_sample_counts; }

  // Opt MC into endpoint-inclusive sampling so the grid lands exactly on the
  // cell faces (frac = 0 and frac = 1 planes).
  inline bool inclusive_endpoints() const { return true; }

  // Map MC's fractional-space gradient to cartesian. Used by extract_with_
  // curvature for normals/curvature.
  inline FMat3 basis_transform() const { return m_normal_transform; }

  inline int num_calls() const { return m_num_calls; }
  inline const auto &molecule() const { return m_molecule; }

private:
  void update_region();

  occ::crystal::Crystal m_crystal;

  float m_buffer{static_cast<float>(DEFAULT_BUFFER_ANGSTROM)};
  InterpolatorParams m_interpolator_params;
  float m_target_separation{0.2f * occ::units::ANGSTROM_TO_BOHR};

  // MC sampling region in fractional coordinates.
  FVec3 m_origin{0.0f, 0.0f, 0.0f};
  FVec3 m_side_length{1.0f, 1.0f, 1.0f};
  Eigen::Vector3i m_sample_counts{1, 1, 1};

  // Lattice matrices in bohr; m_direct_bohr maps frac -> cart(bohr),
  // m_normal_transform = (M^-1)^T maps frac-space gradients to cart-bohr.
  FMat3 m_direct_bohr;
  FMat3 m_normal_transform;

  mutable int m_num_calls{0};

  std::vector<AtomInterpolator> m_atom_interpolators;
  AtomCellList m_cell_list;
  occ::core::Molecule m_molecule;
  ankerl::unordered_dense::map<int, LinearInterpolatorFloat> m_interpolators;
};

} // namespace occ::isosurface
