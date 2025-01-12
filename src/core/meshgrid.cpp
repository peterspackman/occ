#include <occ/core/meshgrid.h>

namespace occ::core {
std::pair<Mat, Mat> meshgrid(const Vec &x, const Vec &y) {
  Mat g0 = x.replicate(1, y.rows()).transpose();
  Mat g1 = y.replicate(1, x.rows());
  return {g0, g1};
}
} // namespace occ::core
