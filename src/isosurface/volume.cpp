#include <occ/isosurface/volume.h>

namespace occ::isosurface {

VolumeGridFunctor::VolumeGridFunctor(VolumeGridPtr grid, float sep,
                                     GridInterpolation interp,
                                     const Eigen::Matrix3f &basis)
    : m_grid(grid), m_target_separation{sep}, m_interpolation(interp),
      m_basis(basis) {
  m_has_transform = !m_basis.isIdentity();
  if (m_has_transform) {
    m_basis_inverse = m_basis.inverse();
  }
}

void VolumeGridFunctor::batch(Eigen::Ref<const FMat3N> pos,
                              Eigen::Ref<FVec> layer) const {
  m_num_calls += layer.size();

  // Transform positions if needed
  FMat3N transformed_pos;
  if (m_has_transform) {
    transformed_pos = m_basis_inverse * pos;
  } else {
    transformed_pos = pos;
  }

  switch (m_interpolation) {
  case GridInterpolation::NearestPoint:
    batch_nearest(transformed_pos, layer);
    break;
  case GridInterpolation::Trilinear:
    batch_trilinear(transformed_pos, layer);
    break;
  }
}

void VolumeGridFunctor::remap_vertices(const std::vector<float> &v,
                                       std::vector<float> &dest) const {
  if (m_has_transform) {
    impl::remap_vertices(*this, v, dest);
  } else {
    dest.resize(v.size());
    Eigen::Map<FMat3N>(dest.data(), 3, v.size() / 3) =
        m_basis * Eigen::Map<const FMat3N>(v.data(), 3, v.size() / 3);
  }
}

void VolumeGridFunctor::batch_nearest(const FMat3N &pos,
                                      Eigen::Ref<FVec> layer) const {
  for (int i = 0; i < pos.cols(); ++i) {
    const auto &p = pos.col(i);

    // Convert to grid indices
    int ix = std::clamp(static_cast<int>(std::round(p.x())), 0,
                        static_cast<int>(m_grid->nx() - 1));
    int iy = std::clamp(static_cast<int>(std::round(p.y())), 0,
                        static_cast<int>(m_grid->ny() - 1));
    int iz = std::clamp(static_cast<int>(std::round(p.z())), 0,
                        static_cast<int>(m_grid->nz() - 1));

    layer(i) = (*m_grid)(ix, iy, iz);
  }
}

void VolumeGridFunctor::batch_trilinear(const FMat3N &pos,
                                        Eigen::Ref<FVec> layer) const {
  for (int i = 0; i < pos.cols(); ++i) {
    const auto &p = pos.col(i);

    // Convert to grid indices with fractional components
    float fx = p.x() * (m_grid->nx() - 1);
    float fy = p.y() * (m_grid->ny() - 1);
    float fz = p.z() * (m_grid->nz() - 1);

    // Get integer indices and weights
    int ix = std::floor(fx);
    int iy = std::floor(fy);
    int iz = std::floor(fz);

    // Clamp to valid range
    ix = std::clamp(ix, 0, static_cast<int>(m_grid->nx() - 2));
    iy = std::clamp(iy, 0, static_cast<int>(m_grid->ny() - 2));
    iz = std::clamp(iz, 0, static_cast<int>(m_grid->nz() - 2));

    float wx = fx - ix;
    float wy = fy - iy;
    float wz = fz - iz;

    // Trilinear interpolation using adjacent slices
    const auto &slice0 = m_grid->slice(ix);
    const auto &slice1 = m_grid->slice(ix + 1);

    float v000 = slice0(iy, iz);
    float v001 = slice0(iy, iz + 1);
    float v010 = slice0(iy + 1, iz);
    float v011 = slice0(iy + 1, iz + 1);
    float v100 = slice1(iy, iz);
    float v101 = slice1(iy, iz + 1);
    float v110 = slice1(iy + 1, iz);
    float v111 = slice1(iy + 1, iz + 1);

    layer(i) = (1 - wx) * (1 - wy) * (1 - wz) * v000 +
               (1 - wx) * (1 - wy) * wz * v001 +
               (1 - wx) * wy * (1 - wz) * v010 + (1 - wx) * wy * wz * v011 +
               wx * (1 - wy) * (1 - wz) * v100 + wx * (1 - wy) * wz * v101 +
               wx * wy * (1 - wz) * v110 + wx * wy * wz * v111;
  }
}

} // namespace occ::isosurface
