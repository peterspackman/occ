#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::core {

// In point_charge.h
class PointCharge {
public:
  PointCharge(double charge, double x, double y, double z)
      : m_charge(charge), m_position(x, y, z) {}

  PointCharge(double charge, const std::array<double, 3> &pos)
      : m_charge(charge), m_position(pos[0], pos[1], pos[2]) {}

  PointCharge(double charge, const Vec3 &pos)
      : m_charge(charge), m_position(pos) {}

  inline double charge() const { return m_charge; }
  inline void set_charge(double q) { m_charge = q; }
  inline const auto &position() const { return m_position; }
  inline void set_position(const Vec3 &pos) { m_position = pos; }

private:
  double m_charge;
  Vec3 m_position;
};

std::vector<PointCharge> inline make_point_charges(
    const std::vector<Atom> &atoms) {
  std::vector<PointCharge> q;
  q.reserve(atoms.size());
  for (const auto &atom : atoms) {
    q.emplace_back(static_cast<double>(atom.atomic_number), atom.x, atom.y,
                   atom.z);
  }
  return q;
}

constexpr unsigned int num_multipole_components(unsigned int order) {
  unsigned int n{0};
  for (unsigned int i = 0; i <= order; i++)
    n += (i + 1) * (i + 2) / 2;
  return n;
}

template <unsigned int order = 0>
inline auto compute_multipoles(const std::vector<PointCharge> &charges,
                               const Vec3 &origin = Vec3::Zero()) {
  constexpr unsigned int ncomp = num_multipole_components(order);

  std::array<double, ncomp> result{0.0};
  for (const auto &pc : charges) {
    const double charge = pc.charge();
    result[0] += charge;
    const Vec3 d = pc.position() - origin;
    const double x = d.x();
    const double y = d.y();
    const double z = d.z();
    if constexpr (order > 0) {
      result[1] += charge * x;
      result[2] += charge * y;
      result[3] += charge * z;
    }
    if constexpr (order > 1) {
      result[4] += charge * x * x; // xx
      result[5] += charge * x * y; // xy
      result[6] += charge * x * z; // xz
      result[7] += charge * y * y; // yy
      result[8] += charge * y * z; // yz
      result[9] += charge * z * z; // zz
    }
    if constexpr (order > 2) {
      result[10] += charge * x * x * x; // xxx
      result[11] += charge * x * x * y; // xxy
      result[12] += charge * x * x * z; // xxz
      result[13] += charge * x * y * y; // xyy
      result[14] += charge * x * y * z; // xyz
      result[15] += charge * x * z * z; // xzz
      result[16] += charge * y * y * y; // yyy
      result[17] += charge * y * y * z; // yyz
      result[18] += charge * y * z * z; // yzz
      result[19] += charge * z * z * z; // zzz
    }
    if constexpr (order > 3) {
      result[20] += charge * x * x * x * x; // xxxx
      result[21] += charge * x * x * x * y; // xxxy
      result[22] += charge * x * x * x * z; // xxxz
      result[23] += charge * x * x * y * y; // xxyy
      result[24] += charge * x * x * y * z; // xxyz
      result[25] += charge * x * x * z * z; // xxzz
      result[26] += charge * x * y * y * y; // xyyy
      result[27] += charge * x * y * y * z; // xyyz
      result[28] += charge * x * y * z * z; // xyzz
      result[29] += charge * x * z * z * z; // xzzz
      result[30] += charge * y * y * y * y; // yyyy
      result[31] += charge * y * y * y * z; // yyyz
      result[32] += charge * y * y * z * z; // yyzz
      result[33] += charge * y * z * z * z; // yzzz
      result[34] += charge * z * z * z * z; // zzzz
    }
  }
  return result;
}

} // namespace occ::core
