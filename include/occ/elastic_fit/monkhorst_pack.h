#include <occ/core/linear_algebra.h>

namespace occ::elastic_fit {

struct MonkhorstPack {
  size_t shrink_x, shrink_y, shrink_z;
  occ::Vec3 shift;
  occ::Mat3N grid;

  inline void generate_grid() {
    size_t n_kpoints = shrink_x * shrink_y * shrink_z;
    grid = occ::Mat3N(3, n_kpoints);
    size_t i_point = 0;
    double x_shift =
        shift[0] ? shift[0] : 1.0 / static_cast<double>(shrink_x) / 2;
    double y_shift =
        shift[1] ? shift[1] : 1.0 / static_cast<double>(shrink_y) / 2;
    double z_shift =
        shift[2] ? shift[2] : 1.0 / static_cast<double>(shrink_z) / 2;
    for (size_t x = 0; x < shrink_x; x++) {
      double x_point = static_cast<double>(x) / static_cast<double>(shrink_x);
      for (size_t y = 0; y < shrink_y; y++) {
        double y_point = static_cast<double>(y) / static_cast<double>(shrink_y);
        for (size_t z = 0; z < shrink_z; z++) {
          double z_point =
              static_cast<double>(z) / static_cast<double>(shrink_z);
          occ::Vec3 point = occ::Vec3(x_point + x_shift, y_point + y_shift,
                                      z_point + z_shift);
          grid.col(i_point) = point;
          i_point++;
        }
      }
    }
  }

  MonkhorstPack(const occ::IVec3 &shrinking_factors, const occ::Vec3 &shift)
      : shrink_x(shrinking_factors[0]), shrink_y(shrinking_factors[1]),
        shrink_z(shrinking_factors[2]), shift(shift) {
    this->generate_grid();
  }

  inline const size_t size() const { return grid.cols(); }

  inline auto begin() const { return grid.colwise().begin(); }
  inline auto end() const { return grid.colwise().end(); }
};

} // namespace occ::elastic_fit
