#pragma once
#include <occ/geometry/volume_grid.h>
#include <occ/isosurface/common.h>

namespace occ::isosurface {

enum class GridInterpolation { NearestPoint, Trilinear };

using VolumeGridPtr = std::shared_ptr<const occ::geometry::VolumeGrid>;

class VolumeGridFunctor {
public:
  VolumeGridFunctor(VolumeGridPtr grid, float sep,
                    GridInterpolation interp = GridInterpolation::NearestPoint,
                    const Eigen::Matrix3f &basis = Eigen::Matrix3f::Identity());

  void batch(Eigen::Ref<const FMat3N> pos, Eigen::Ref<FVec> layer) const;

  void remap_vertices(const std::vector<float> &v,
                      std::vector<float> &dest) const;

  inline int num_calls() const { return m_num_calls; }

  inline void set_basis(const Eigen::Matrix3f &basis) {
    m_basis = basis;
    m_has_transform = !m_basis.isIdentity();
    if (m_has_transform) {
      m_basis_inverse = m_basis.inverse();
    }
  }

  void set_interpolation(GridInterpolation interp) { m_interpolation = interp; }

  GridInterpolation interpolation() const { return m_interpolation; }

  inline const auto &side_length() const { return m_cube_side_length; }

  inline Eigen::Vector3i cubes_per_side() const {
    return (side_length().array() / m_target_separation).ceil().cast<int>();
  }

  inline const auto &origin() const { return m_origin; }

  inline void update_num_calls(int n) const { m_num_calls += n; }

private:
  void batch_nearest(const FMat3N &pos, Eigen::Ref<FVec> layer) const;
  void batch_trilinear(const FMat3N &pos, Eigen::Ref<FVec> layer) const;

  mutable int m_num_calls{0};
  VolumeGridPtr m_grid;
  float m_target_separation{0.2};
  GridInterpolation m_interpolation;
  bool m_has_transform{false};
  Eigen::Matrix3f m_basis;
  Eigen::Matrix3f m_basis_inverse;
  Eigen::Vector3f m_cube_side_length;
  Eigen::Vector3f m_origin;
};

} // namespace occ::isosurface
