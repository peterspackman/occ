#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/crystal/hkl.h>
#include <string>

namespace occ::crystal {
using occ::util::is_close;

/**
 * This class represents a unit cell of a crystal lattice.
 *
 * A UnitCell describes the lattice vectors of a 3D crystal
 * structure, including the 6 (possibly unique) parameters
 * lengths \f$(a,b,c)\f$ and angles \f$(\alpha, \beta, \gamma)\f$.
 * It also serves as a utility class for coordinate transforms,
 * and other conversions and checks.
 */

class UnitCell {
public:
  UnitCell();
  UnitCell(double a, double b, double c, double alpha, double beta,
           double gamma);
  UnitCell(const Vec3 &lengths, const Vec3 &angles);
  UnitCell(const Mat3 &vectors);

  /// the length of the a-axis in Angstroms
  inline double a() const { return m_lengths(0); }
  /// the length of the b-axis in Angstroms
  inline double b() const { return m_lengths(1); }
  /// the length of the c-axis in Angstroms
  inline double c() const { return m_lengths(2); }

  /// angle \f$\alpha\f$, i.e. the angle between the b- and c-axes in radians
  inline double alpha() const { return m_angles(0); }
  /// angle \f$\beta\f$, i.e. the angle between the a- and c-axes in radians
  inline double beta() const { return m_angles(1); }
  /// angle \f$\gamma\f$, i.e. the angle between the a- and b-axes in radians
  inline double gamma() const { return m_angles(2); }

  /// Volume of the unit cell in cubic Angstroms
  inline double volume() const { return m_volume; }

  inline Mat3 metric_tensor() const { return m_direct.transpose() * m_direct; }

  inline Mat3 reciprocal_metric_tensor() const {
    return m_reciprocal.transpose() * m_reciprocal;
  }

  /**
   * Set the length of the a-axis.
   *
   * \param a new length of the a-axis in Angstroms
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_a(double a);

  /**
   * Set the length of the b-axis.
   *
   * \param b new length of the b-axis in Angstroms
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_b(double b);

  /**
   * Set the length of the c-axis.
   *
   * \param c new length of the c-axis in Angstroms
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_c(double c);

  /**
   * Set the angle of between the b- and c-axes
   *
   * \param alpha new angle between the b- and c-axes in radians
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_alpha(double alpha);

  /**
   * Set the angle of between the a- and c-axes
   *
   * \param beta new angle between the a- and c-axes in radians
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_beta(double beta);

  /**
   * Set the angle of between the a- and b-axes
   *
   * \param gamma new angle between the a- and b-axes in radians
   *
   * \note Will update other dependent values e.g. volume etc.
   * No checking is performed.
   */
  void set_gamma(double gamma);

  /// Check if cubic i.e. \f$ a=b=c, \alpha=\beta=\gamma \f$
  bool is_cubic() const;

  /// Check if triclinic i.e. \f$ a \ne b \ne c, \alpha \ne \beta \ne \gamma
  /// \f$
  bool is_triclinic() const;

  /// Check if monoclinic i.e. \f$ a \ne b \ne c, \alpha = \gamma = 90^\circ
  /// \ne \beta \f$
  bool is_monoclinic() const;

  /// Check if orthorhombic i.e. \f$ a \ne b \ne c, \alpha = \beta = \gamma =
  /// 90^\circ \f$
  bool is_orthorhombic() const;

  /// Check if tetragonal i.e. \f$ a = b, a \ne c, \alpha = \beta = \gamma =
  /// 90^\circ \f$
  bool is_tetragonal() const;

  /// Check if rhombohedral i.e. \f$ a = b = c, \alpha = \beta = \gamma \ne
  /// 90^\circ \f$
  bool is_rhombohedral() const;

  /// Check if hexagonal i.e. \f$ a = b, a \ne c, \alpha = 90^\circ , \gamma =
  /// 120^\circ \f$
  bool is_hexagonal() const;

  /// Check if orthogonal i.e. \f$\alpha = \beta = \gamma = 90^\circ\f$
  inline bool is_orthogonal() const { return _a_90() && _b_90() && _c_90(); }

  /// String representing the type of cell e.g. "monoclinic"
  std::string cell_type() const;

  /**
   * Convert a given matrix of coordinates from fractional to Cartesian
   *
   * \param coords Mat3N of fractional coordinates
   *
   * \returns Mat3N of Cartesian coordinates
   */
  inline auto to_cartesian(const Mat3N &coords) const {
    return m_direct * coords;
  }

  /**
   * Calculate the direct matrix for ADP transformations
   *
   * This method computes the ad hoc direct matrix used specifically
   * for transforming Anisotropic Displacement Parameters (ADPs).
   *
   * \returns Mat3 representing the ADP direct transformation matrix
   */
  Mat3 adp_adhoc_direct() const;

  /**
   * Calculate the inverse matrix for ADP transformations
   *
   * This method computes the ad hoc inverse matrix used specifically
   * for transforming Anisotropic Displacement Parameters (ADPs).
   *
   * \returns Mat3 representing the ADP inverse transformation matrix
   */
  Mat3 adp_adhoc_inverse() const;

  /**
   * Convert ADPs from Cartesian to fractional coordinates
   *
   * This method transforms a set of Anisotropic Displacement Parameters (ADPs)
   * from Cartesian to the fractional (ad-hoc) coordinate system.
   *
   * \param adp Mat6N of ADPs in Cartesian coordinates
   * \returns Mat6N of ADPs in fractional coordinates
   */
  Mat6N to_fractional_adp(const Mat6N &adp) const;

  /**
   * Convert ADPs from fractional to Cartesian coordinates
   *
   * This method transforms a set of Anisotropic Displacement Parameters (ADPs)
   * from fractional (ad-hoc) basis to Cartesian coordinate system.
   *
   * \param adp Mat6N of ADPs in fractional coordinates
   * \returns Mat6N of ADPs in Cartesian coordinates
   */
  Mat6N to_cartesian_adp(const Mat6N &adp) const;

  /**
   * Convert a given matrix of coordinates from Cartesian to fractional
   *
   * \param coords Mat3N of Cartesian coordinates
   *
   * \returns Mat3N of fractional coordinates
   */
  inline auto to_fractional(const Mat3N &coords) const {
    return m_inverse * coords;
  }

  /**
   * Convert a given matrix of coordinates from Cartesian to fractional
   *
   * \param coords Mat3N of Cartesian coordinates
   *
   * \returns Mat3N of fractional coordinates
   */
  inline auto to_reciprocal(const Mat3N &coords) const {
    return m_reciprocal * coords;
  }

  /// The direct matrix of this unit cell (columns are lattice vectors)
  inline const auto &direct() const { return m_direct; }
  /// The reciprocal matrix of this unit cell (columns are reciprocal lattice
  /// vectors)
  inline const auto &reciprocal() const { return m_reciprocal; }
  /// The inverse matrix of this unit cell (rows are reciprocal lattice
  /// vectors)
  inline const auto &inverse() const { return m_inverse; }

  /// the \f$\bf{a}\f$ lattice vector
  inline auto a_vector() const { return m_direct.col(0); }
  /// the \f$\bf{b}\f$ lattice vector
  inline auto b_vector() const { return m_direct.col(1); }
  /// the \f$\bf{c}\f$ lattice vector
  inline auto c_vector() const { return m_direct.col(2); }

  /// the \f$\bf{a^*}\f$ vector in reciprocal lattice
  inline auto a_star_vector() const { return m_reciprocal.col(0); }
  /// the \f$\bf{b^*}\f$ vector in reciprocal lattice
  inline auto b_star_vector() const { return m_reciprocal.col(1); }
  /// the \f$\bf{c^*}\f$ vector in reciprocal lattice
  inline auto c_star_vector() const { return m_reciprocal.col(2); }

  /// Vector of lengths \f$(a, b, c)\f$
  inline const auto &lengths() const { return m_lengths; }

  /// Vector of lengths \f$(a, b, c)\f$
  inline const auto &angles() const { return m_angles; }

  /**
   * Return the maximum fractional coordinates \f$(h,k,l)\f$ needed
   * to enclose a sphere of size d_min
   *
   * \return HKL of the limits bounding a given sphere
   */
  HKL hkl_limits(double d_min) const;

private:
  void update_cell_matrices();
  inline bool _ab_close() const { return is_close(m_lengths(0), m_lengths(1)); }
  inline bool _ac_close() const { return is_close(m_lengths(0), m_lengths(2)); }
  inline bool _bc_close() const { return is_close(m_lengths(1), m_lengths(2)); }
  inline bool _abc_close() const {
    return _ab_close() && _ac_close() && _bc_close();
  }
  inline bool _abc_different() const {
    return !_ab_close() && !_ac_close() && !_bc_close();
  }
  inline bool _a_ab_close() const { return is_close(m_angles(0), m_angles(1)); }
  inline bool _a_ac_close() const { return is_close(m_angles(0), m_angles(2)); }
  inline bool _a_bc_close() const { return is_close(m_angles(1), m_angles(2)); }
  inline bool _a_abc_close() const {
    return _a_ab_close() && _a_ac_close() && _a_bc_close();
  }
  inline bool _a_abc_different() const {
    return !_a_ab_close() && !_a_ac_close() && !_a_bc_close();
  }
  inline bool _a_90() const {
    return is_close(m_angles(0), occ::units::PI / 2);
  }
  inline bool _b_90() const {
    return is_close(m_angles(1), occ::units::PI / 2);
  }
  inline bool _c_90() const {
    return is_close(m_angles(2), occ::units::PI / 2);
  }

  Vec3 m_lengths;
  Vec3 m_angles;
  Vec3 m_sin;
  Vec3 m_cos;
  double m_volume = 0.0;

  Mat3 m_direct;
  Mat3 m_inverse;
  Mat3 m_reciprocal;
};

/// Construct a cubic unit cell from a given length
UnitCell cubic_cell(double length);
/// Construct a rhombohedral unit cell from a given length and angle
UnitCell rhombohedral_cell(double length, double angle);
/// Construct a tetragonal unit cell from two given lengths
UnitCell tetragonal_cell(double a, double c);
/// Construct a hexagonal unit cell from two given lengths
UnitCell hexagonal_cell(double a, double c);
/// Construct an orthorhombic unit cell from three given lengths
UnitCell orthorhombic_cell(double a, double b, double c);
/// Construct an orthorhombic unit cell from three given lengths and one angle
UnitCell monoclinic_cell(double a, double b, double c, double angle);
/// Construct an orthorhombic unit cell from three given lengths and three
/// angles
UnitCell triclinic_cell(double a, double b, double c, double alpha, double beta,
                        double gamma);

} // namespace occ::crystal
