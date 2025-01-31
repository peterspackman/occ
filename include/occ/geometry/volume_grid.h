#include <array>
#include <memory>
#include <occ/core/linear_algebra.h>

namespace occ::geometry {

class VolumeGrid {
public:
  VolumeGrid(size_t x, size_t y, size_t z);
  VolumeGrid(const std::array<size_t, 3> &dims);

  VolumeGrid(std::unique_ptr<float[]> buffer,
             const std::array<size_t, 3> &dims);
  VolumeGrid(VolumeGrid &&other) = default;
  VolumeGrid &operator=(VolumeGrid &&other) = default;

  float &operator()(size_t x, size_t y, size_t z);
  const float &operator()(size_t x, size_t y, size_t z) const;

  Eigen::Map<FMatRM> slice(size_t x);
  Eigen::Map<const FMatRM> slice(size_t x) const;

  void set_zero();
  inline const std::array<size_t, 3> &dimensions() const { return m_dims; }

  inline size_t nx() const { return m_dims[0]; }
  inline size_t ny() const { return m_dims[1]; }
  inline size_t nz() const { return m_dims[2]; }

  inline size_t size() const { return nx() * ny() * nz(); }

  inline float *data() { return m_data.get(); }
  inline const float *data() const { return m_data.get(); }

private:
  std::array<size_t, 3> m_dims;
  std::unique_ptr<float[]> m_data;
};

} // namespace occ::geometry
