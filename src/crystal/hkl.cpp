#include <occ/crystal/hkl.h>

namespace occ::crystal {

double HKL::d(const Mat3 &lattice) const {
  return Vec3(h * lattice.col(0) + k * lattice.col(1) + l * lattice.col(2))
      .norm();
}

Vec3 HKL::vector() const {
  return Vec3(static_cast<double>(h), static_cast<double>(k),
              static_cast<double>(l));
}

HKL HKL::floor(const Vec3 &vec, double epsilon) {
  HKL r;
  r.h = static_cast<int>(std::floor(vec(0) + epsilon));
  r.k = static_cast<int>(std::floor(vec(1) + epsilon));
  r.l = static_cast<int>(std::floor(vec(2) + epsilon));
  return r;
}

HKL HKL::ceil(const Vec3 &vec) {
  HKL r;
  r.h = static_cast<int>(std::ceil(vec(0)));
  r.k = static_cast<int>(std::ceil(vec(1)));
  r.l = static_cast<int>(std::ceil(vec(2)));
  return r;
}

HKL HKL::from_vector(const Vec3 &vec) {
  HKL r;
  r.h = static_cast<int>(vec(0));
  r.k = static_cast<int>(vec(1));
  r.l = static_cast<int>(vec(2));
  return r;
}

} // namespace occ::crystal

auto fmt::formatter<occ::crystal::HKL>::format(const occ::crystal::HKL &hkl,
                                               format_context &ctx) const
    -> decltype(ctx.out()) {
  return fmt::format_to(ctx.out(), "HKL[{} {} {}]", (hkl.h), nested(hkl.k),
                        nested(hkl.l));
}
