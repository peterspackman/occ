#pragma once
#include <Eigen/Geometry>
#include <occ/core/linear_algebra.h>

namespace occ::core {

/**
 * Class for computation of properties based on an elasticity tensor
 *
 * This implementation would not have been possible without the
 * ELATE[1] software, https://progs.coudert.name/elate
 * which heavily inspired it and was wonderful as a refereence implementation
 * for the various properties.
 *
 * [1] Gaillac et al. https://doi.org/10.1088/0953-8984/28/27/275201
 *
 */
class ElasticTensor {
public:
  enum class AveragingScheme { Voigt, Reuss, Hill, Numerical };

  using AngularDirection = Eigen::Ref<const Eigen::Vector<double, 2>>;
  using CartesianDirection = Eigen::Ref<const Eigen::Vector<double, 3>>;

  explicit ElasticTensor(Eigen::Ref<const Mat6> c_voigt);

  double youngs_modulus_angular(AngularDirection) const;
  double youngs_modulus(CartesianDirection) const;

  double linear_compressibility_angular(AngularDirection) const;
  double linear_compressibility(CartesianDirection) const;

  double shear_modulus_angular(AngularDirection, double angle) const;
  double shear_modulus(CartesianDirection, double angle) const;
  double shear_modulus(CartesianDirection, CartesianDirection) const;
  std::pair<double, double> shear_modulus_minmax(CartesianDirection) const;

  double poisson_ratio_angular(AngularDirection, double angle) const;
  double poisson_ratio(CartesianDirection, double angle) const;
  double poisson_ratio(CartesianDirection, CartesianDirection) const;
  std::pair<double, double> poisson_ratio_minmax(CartesianDirection) const;
  double average_poisson_ratio_direction(CartesianDirection,
                                         int num_samples = 360) const;
  double reduced_youngs_modulus(CartesianDirection,
                                int num_samples = 360) const;

  double
  average_bulk_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_shear_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_youngs_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_poisson_ratio(AveragingScheme avg = AveragingScheme::Hill) const;

  /**
   * Calculate the transverse (shear) acoustic velocity.
   *
   * \param bulk_modulus_gpa Bulk modulus in GPa (not used in calculation,
   * included for API consistency)
   * \param shear_modulus_gpa Shear modulus in GPa
   * \param density_g_cm3 Density in g/cm³
   * \returns Transverse acoustic velocity in m/s
   *
   * Calculates V_s = sqrt(G/ρ) where G is the shear modulus and ρ is the
   * density. This represents the velocity of shear waves propagating through
   * the material.
   */
  double transverse_acoustic_velocity(double bulk_modulus_gpa,
                                      double shear_modulus_gpa,
                                      double density_g_cm3) const;

  /**
   * Calculate the longitudinal (compressional) acoustic velocity.
   *
   * \param bulk_modulus_gpa Bulk modulus in GPa
   * \param shear_modulus_gpa Shear modulus in GPa
   * \param density_g_cm3 Density in g/cm³
   * \returns Longitudinal acoustic velocity in m/s
   *
   * Calculates V_p = sqrt((4G + 3K)/(3ρ)) where G is the shear modulus,
   * K is the bulk modulus, and ρ is the density.
   * This represents the velocity of compressional waves propagating through the
   * material.
   */
  double longitudinal_acoustic_velocity(double bulk_modulus_gpa,
                                        double shear_modulus_gpa,
                                        double density_g_cm3) const;

  const Mat6 &voigt_s() const;
  const Mat6 &voigt_c() const;

  /**
   * Get the 6x6 rotation matrix in order to rotate the Voigt representations of
   * the tensors
   *
   * \param rotation 3x3 rotation matrix R
   * \returns 6x6 rotation matrix Tto be used as (C' = T @ C @ T^T)
   */
  Mat6 voigt_rotation_matrix(const Mat3 &rotation) const;

  /**
   * Rotate the elastic tensor using a 3x3 rotation matrix.
   *
   * \param rotation 3x3 rotation matrix R
   * \returns Rotated 6x6 stiffness tensor in Voigt notation (C' = T @ C @ T^T)
   */
  Mat6 rotate_voigt_stiffness(const Mat3 &rotation) const;

  /**
   * Rotate the compliance tensor using a 3x3 rotation matrix.
   *
   * \param rotation 3x3 rotation matrix R
   * \returns Rotated 6x6 compliance tensor in Voigt notation (S' = T @ S @ T^T)
   */
  Mat6 rotate_voigt_compliance(const Mat3 &rotation) const;

  double component(int i, int j, int k, int l) const;
  double &component(int i, int j, int k, int l);

  Vec6 eigenvalues() const;

  inline double *data() { return &m_components[0][0][0][0]; }
  inline double const *data() const { return &m_components[0][0][0][0]; }

private:
  Mat6 m_c;
  Mat6 m_s;
  double m_components[3][3][3][3];
};

} // namespace occ::core
