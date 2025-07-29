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
    
    int num_threads = occ::parallel::get_num_threads();
    // Chunk by z-slices for better cache locality
    int z_per_thread = (steps(2) + num_threads - 1) / num_threads;
    
    auto inner_func = [&](int thread_id) {
      int z_start = thread_id * z_per_thread;
      int z_end = std::min(z_start + z_per_thread, steps(2));
      if (z_start >= steps(2)) return;
      
      size_t chunk_size = steps(0) * steps(1) * (z_end - z_start);
      Mat3N points(3, chunk_size);
      Vec temp = Vec::Zero(chunk_size);
      
      // Generate points for this z-slice chunk
      size_t local_idx = 0;
      for (int z = z_start; z < z_end; z++) {
        for (int y = 0; y < steps(1); y++) {
          for (int x = 0; x < steps(0); x++, local_idx++) {
            points.col(local_idx) = basis * Vec3(x, y, z) + origin;
          }
        }
      }
      
      // Process chunk
      func(points, temp);
      
      // Copy results back to grid - calculate proper offset
      size_t grid_offset = z_start * steps(0) * steps(1);
      std::copy(temp.data(), temp.data() + chunk_size, m_grid.data() + grid_offset);
    };
    
    occ::parallel::parallel_do(inner_func);
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
