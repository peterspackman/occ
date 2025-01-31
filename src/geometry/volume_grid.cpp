#include <occ/geometry/volume_grid.h>

namespace occ::geometry {

VolumeGrid::VolumeGrid(size_t x, size_t y, size_t z)
    : m_dims{x, y, z}, m_data(std::make_unique<float[]>(size())) {}

VolumeGrid::VolumeGrid(const std::array<size_t, 3> &dims)
    : m_dims{dims}, m_data(std::make_unique<float[]>(size())) {}

VolumeGrid::VolumeGrid(std::unique_ptr<float[]> buffer, const std::array<size_t, 3> &dims)
    : m_dims{dims}, m_data(std::move(buffer)) {}

float &VolumeGrid::operator()(size_t x, size_t y, size_t z) {
  return m_data[z + y * nz() + x * nz() * ny()];
}

const float &VolumeGrid::operator()(size_t x, size_t y, size_t z) const {
  return m_data[z + y * nz() + x * nz() * ny()];
}

Eigen::Map<FMatRM> VolumeGrid::slice(size_t x) {
  return Eigen::Map<FMatRM>(&m_data[x * ny() * nz()], ny(), nz());
}

Eigen::Map<const FMatRM> VolumeGrid::slice(size_t x) const {
  return Eigen::Map<const FMatRM>(&m_data[x * ny() * nz()], ny(), nz());
}

void VolumeGrid::set_zero() {
  std::memset(m_data.get(), 0, size() * sizeof(float));
}

} // namespace occ::geometry
