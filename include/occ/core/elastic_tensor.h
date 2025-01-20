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

  double
  average_bulk_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_shear_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_youngs_modulus(AveragingScheme avg = AveragingScheme::Hill) const;
  double
  average_poisson_ratio(AveragingScheme avg = AveragingScheme::Hill) const;

  const Mat6 &voigt_s() const;
  const Mat6 &voigt_c() const;

  double component(int i, int j, int k, int l) const;
  double &component(int i, int j, int k, int l);

  inline double *data() { return &m_components[0][0][0][0]; }
  inline double const *data() const { return &m_components[0][0][0][0]; }

private:
  Mat6 m_c;
  Mat6 m_s;
  double m_components[3][3][3][3];
};

} // namespace occ::core
