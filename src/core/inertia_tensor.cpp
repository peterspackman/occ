#include <occ/core/inertia_tensor.h>
#include <occ/core/log.h>

namespace occ::core {

Mat3 inertia_tensor(Eigen::Ref<const Vec> masses,
                    Eigen::Ref<const Mat3N> positions) {
  Mat3 result;
  double total_mass = masses.array().sum();
  Vec3 center_of_mass =
      (positions.array().rowwise() * masses.transpose().array())
          .rowwise()
          .sum() /
      total_mass;
  Mat3N d = positions.colwise() - center_of_mass;
  Mat3N md = d.array().rowwise() * masses.transpose().array();
  Mat3N d2 = d.array() * d.array();
  Mat3N md2 = d2.array().rowwise() * masses.transpose().array();

  result(0, 0) = (md2.row(1).array() + md2.row(2).array()).array().sum();
  result(1, 1) = (md2.row(0).array() + md2.row(2).array()).array().sum();
  result(2, 2) = (md2.row(0).array() + md2.row(1).array()).array().sum();
  result(0, 1) = -md.row(0).dot(d.row(1));
  result(1, 0) = result(0, 1);
  result(0, 2) = -md.row(0).dot(d.row(2));
  result(2, 0) = result(0, 2);
  result(1, 2) = -md.row(1).dot(d.row(2));
  result(2, 1) = result(1, 2);

  return result;
}

} // namespace occ::core
