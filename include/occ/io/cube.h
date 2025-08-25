#pragma once
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/geometry/volume_grid.h>
#include <vector>

namespace occ::io {
class Cube {
public:
  Cube();
  using AtomList = std::vector<core::Atom>;

  template <typename F> void fill_data_from_function(F &func) {
    // Create volume grid with appropriate dimensions
    m_grid = geometry::VolumeGrid(steps(0), steps(1), steps(2));
    
    // Use TBB thread-local storage for efficient memory reuse
    occ::parallel::thread_local_storage<Mat3N> tl_points;
    occ::parallel::thread_local_storage<Vec> tl_temp;
    
    // Let TBB handle work distribution with automatic load balancing
    // Process z-slices individually for optimal granularity and cache locality
    occ::parallel::parallel_for(0, steps(2), [&](int z) {
      const size_t slice_size = steps(0) * steps(1);
      
      // Get thread-local storage, resize if needed
      auto& points = tl_points.local();
      auto& temp = tl_temp.local();
      
      if (points.cols() < slice_size) {
        points.resize(3, slice_size);
      }
      if (temp.size() < slice_size) {
        temp.resize(slice_size);
      }
      
      // Generate points for this z-slice
      size_t local_idx = 0;
      for (int y = 0; y < steps(1); y++) {
        for (int x = 0; x < steps(0); x++, local_idx++) {
          points.col(local_idx) = basis * Vec3(x, y, z) + origin;
        }
      }
      
      // Process slice (only use the needed portion)
      auto points_view = points.leftCols(slice_size);
      auto temp_view = temp.head(slice_size);
      temp_view.setZero();
      func(points_view, temp_view);
      
      // Copy results back to grid
      size_t grid_offset = z * steps(0) * steps(1);
      std::copy(temp_view.data(), temp_view.data() + slice_size, m_grid.data() + grid_offset);
    });
  }

  void center_molecule();
  void save(const std::string &);
  void save(std::ostream &);

  std::string name{"cube file from OCC"};
  std::string description{"cube file from OCC"};
  Vec3 origin{0, 0, 0};
  Mat3 basis;
  IVec3 steps{11, 11, 11};
  AtomList atoms;
  Vec charges;

  // Accessor for data
  const float *data() const { return m_grid.data(); }
  float *data() { return m_grid.data(); }

  // Get the underlying VolumeGrid
  const geometry::VolumeGrid &grid() const { return m_grid; }
  geometry::VolumeGrid &grid() { return m_grid; }

  static Cube load(const std::string &filename);
  static Cube load(std::istream &input);

private:
  void write_data_to_stream(std::ostream &);

  void write_header_to_stream(std::ostream &);
  geometry::VolumeGrid m_grid;
};
} // namespace occ::io
