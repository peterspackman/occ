#pragma once
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/geometry/volume_grid.h>
#include <vector>

namespace occ::io {
class Cube {
public:
  Cube();
  using AtomList = std::vector<core::Atom>;

  template <typename F> void fill_data_from_function(F &func) {
    size_t N = steps(0) * steps(1) * steps(2);
    Mat3N points(3, N);

    // Create volume grid with appropriate dimensions
    m_grid = geometry::VolumeGrid(steps(0), steps(1), steps(2));

    for (int x = 0, i = 0; x < steps(0); x++) {
      for (int y = 0; y < steps(1); y++) {
        for (int z = 0; z < steps(2); z++, i++) {
          points.col(i) = basis * Vec3(x, y, z) + origin;
        }
      }
    }

    // Create temporary vector for func output
    Vec temp = Vec::Zero(N);
    func(points, temp);

    // Copy data into volume grid
    std::copy(temp.data(), temp.data() + N, m_grid.data());
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
