#pragma once

#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <string>

namespace occ::crystal {
using occ::Mat3;
using occ::Mat3N;
using occ::Vec3;
using occ::util::isclose;

class UnitCell {
public:
  UnitCell(double a, double b, double c, double alpha, double beta,
           double gamma);
  UnitCell(const Vec3 &lengths, const Vec3 &angles);

  double a() const { return m_lengths[0]; }
  double b() const { return m_lengths[1]; }
  double c() const { return m_lengths[2]; }
  double alpha() const { return m_angles[0]; }
  double beta() const { return m_angles[1]; }
  double gamma() const { return m_angles[2]; }
  void set_a(double);
  void set_b(double);
  void set_c(double);
  void set_alpha(double);
  void set_beta(double);
  void set_gamma(double);

  bool is_cubic() const;
  bool is_triclinic() const;
  bool is_monoclinic() const;
  bool is_orthorhombic() const;
  bool is_tetragonal() const;
  bool is_rhombohedral() const;
  bool is_hexagonal() const;
  bool is_orthogonal() const { return _a_90() && _b_90() && _c_90(); }
  std::string cell_type() const;

  auto to_cartesian(const Mat3N &coords) const { return m_direct * coords; }
  auto to_fractional(const Mat3N &coords) const { return m_inverse * coords; }

  const auto &direct() const { return m_direct; }
  const auto &reciprocal() const { return m_reciprocal; }
  const auto &inverse() const { return m_inverse; }
  auto a_vector() const { return m_direct.col(0); }
  auto b_vector() const { return m_direct.col(1); }
  auto c_vector() const { return m_direct.col(2); }
  auto a_star_vector() const { return m_reciprocal.col(0); }
  auto b_star_vector() const { return m_reciprocal.col(1); }
  auto c_star_vector() const { return m_reciprocal.col(2); }
  const auto& lengths() const { return m_lengths; }

private:
  void update_cell_matrices();
  inline bool _ab_close() const { return isclose(m_lengths(0), m_lengths(1)); }
  inline bool _ac_close() const { return isclose(m_lengths(0), m_lengths(2)); }
  inline bool _bc_close() const { return isclose(m_lengths(1), m_lengths(2)); }
  inline bool _abc_close() const {
    return _ab_close() && _ac_close() && _bc_close();
  }
  inline bool _abc_different() const {
    return !_ab_close() && !_ac_close() && !_bc_close();
  }
  inline bool _a_ab_close() const { return isclose(m_angles(0), m_angles(1)); }
  inline bool _a_ac_close() const { return isclose(m_angles(0), m_angles(2)); }
  inline bool _a_bc_close() const { return isclose(m_angles(1), m_angles(2)); }
  inline bool _a_abc_close() const {
    return _a_ab_close() && _a_ac_close() && _a_bc_close();
  }
  inline bool _a_abc_different() const {
    return !_a_ab_close() && !_a_ac_close() && !_a_bc_close();
  }
  inline bool _a_90() const { return isclose(m_angles(0), M_PI / 2); }
  inline bool _b_90() const { return isclose(m_angles(1), M_PI / 2); }
  inline bool _c_90() const { return isclose(m_angles(2), M_PI / 2); }

  Vec3 m_lengths;
  Vec3 m_angles;
  Vec3 m_sin;
  Vec3 m_cos;
  double m_volume = 0.0;

  Mat3 m_direct;
  Mat3 m_inverse;
  Mat3 m_reciprocal;
};


UnitCell cubic_cell(double length);
UnitCell rhombohedral_cell(double length, double angle);
UnitCell tetragonal_cell(double a, double c);
UnitCell hexagonal_cell(double a, double c);
UnitCell orthorhombic_cell(double a, double b, double c);
UnitCell monoclinic_cell(double a, double b, double c, double angle);
UnitCell triclinic_cell(double a, double b, double c, double alpha, double beta, double gamma);


} // namespace occ::crystal
