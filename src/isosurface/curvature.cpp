#include <occ/core/log.h>
#include <occ/isosurface/curvature.h>

namespace occ::isosurface {

FVec make_vertex_curvedness(const FVec &k1, const FVec &k2) {
  FVec curvedness = (0.5 * (k1.array().square() + k2.array().square())).sqrt();

  // Set zero values to the smallest non-zero value
  float small = std::numeric_limits<float>::max();
  for (int i = 0; i < curvedness.size(); i++) {
    if (curvedness(i) != 0.0f && curvedness(i) < small) {
      small = curvedness(i);
    }
  }
  for (int i = 0; i < curvedness.size(); i++) {
    if (curvedness(i) == 0.0f) {
      curvedness(i) = small;
    }
  }

  // Apply logarithmic scaling
  float fac = 2.0f / M_PI;
  curvedness = fac * curvedness.array().log();

  return curvedness;
}

SurfaceCurvature calculate_curvature(const std::vector<float> &mean,
                                     const std::vector<float> &gaussian) {
  return calculate_curvature(
      Eigen::Map<const FVec>(mean.data(), mean.size()),
      Eigen::Map<const FVec>(gaussian.data(), gaussian.size()));
}

SurfaceCurvature calculate_curvature(const FVec &mean, const FVec &gaussian) {
  SurfaceCurvature result;
  result.mean = Eigen::Map<const FVec>(mean.data(), mean.size());
  result.gaussian = Eigen::Map<const FVec>(gaussian.data(), gaussian.size());
  result.k1 = FVec(mean.size());
  result.k2 = result.k1;
  result.shape_index = result.k1;

  for (int i = 0; i < result.mean.size(); i++) {
    float delta = std::sqrt(
        std::max(result.mean(i) * result.mean(i) - result.gaussian(i), 0.0f));
    result.k1(i) = result.mean(i) + delta;
    result.k2(i) = result.mean(i) - delta;
  }

  // Calculate curvedness using the make_vertex_curvedness function
  result.curvedness = make_vertex_curvedness(result.k1, result.k2);

  // Calculate shape index
  for (int i = 0; i < result.mean.size(); i++) {
    if (result.k1(i) == result.k2(i)) {
      result.shape_index(i) = (result.k1(i) < 0) ? 1.0 : -1.0;
    } else {
      float mx = std::max(result.k1(i), result.k2(i));
      float mn = std::min(result.k1(i), result.k2(i));
      result.shape_index(i) = -2.0 / M_PI * std::atan((mx + mn) / (mx - mn));
    }
  }
  occ::log::debug("Shape index range: {} {} {}", result.shape_index.minCoeff(),
                  result.shape_index.mean(), result.shape_index.maxCoeff());

  return result;
}

} // namespace occ::isosurface
