#include <occ/core/elastic_tensor.h>
#include <occ/core/optimize.h>

namespace occ::core {

Vec3 shear_direction(const Vec3 &axis, double angle) {
  // Ensure the axis is normalized
  Vec3 normalized_axis = axis.normalized();

  // Create a quaternion from the axis-angle representation
  Eigen::Quaterniond q(Eigen::AngleAxisd(angle, normalized_axis));

  // Create a vector perpendicular to the axis
  Vec3 perpendicular = normalized_axis.unitOrthogonal();

  // Rotate the perpendicular vector using the quaternion
  return q * perpendicular;
}

// Function to convert spherical coordinates to Cartesian
Vec3 spherical_to_cartesian(double theta, double phi) {
  return Vec3(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
              std::cos(theta));
}

// Function to calculate shear directions for shear modulus calculation
std::pair<Vec3, Vec3> shear_directions(double theta, double phi, double chi) {
  Vec3 axis = spherical_to_cartesian(theta, phi);
  Vec3 shear_dir = shear_direction(axis, chi);
  return {axis, shear_dir};
}

ElasticTensor::ElasticTensor(Eigen::Ref<const Mat6> c_voigt)
    : m_c(c_voigt), m_s(c_voigt.inverse()) {

  Eigen::Matrix<int, 3, 3> vm;
  vm << 0, 5, 4, 5, 1, 3, 4, 3, 2;

  // Helper function to calculate the coefficient
  auto sv_coeff = [](int p, int q) {
    return 1.0 / ((1 + p / 3) * (1 + q / 3));
  };

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          const int p = vm(i, j);
          const int q = vm(k, l);
          m_components[i][j][k][l] = sv_coeff(p, q) * m_s(p, q);
        }
      }
    }
  }
}

double ElasticTensor::youngs_modulus_angular(AngularDirection dir) const {
  return youngs_modulus(spherical_to_cartesian(dir[0], dir[1]));
}

double ElasticTensor::youngs_modulus(CartesianDirection dir) const {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          sum += m_components[i][j][k][l] * dir(i) * dir(j) * dir(k) * dir(l);
        }
      }
    }
  }
  return 1.0 / sum;
}

double
ElasticTensor::linear_compressibility_angular(AngularDirection dir) const {
  return linear_compressibility(spherical_to_cartesian(dir(0), dir(1)));
}

double ElasticTensor::linear_compressibility(CartesianDirection dir) const {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        sum += m_components[i][j][k][k] * dir(i) * dir(j);
      }
    }
  }
  return 1000.0 * sum;
}

double ElasticTensor::shear_modulus_angular(AngularDirection dir,
                                            double angle) const {
  auto [axis, shear_dir] = shear_directions(dir[0], dir[1], angle);
  return shear_modulus(axis, shear_dir);
}

double ElasticTensor::shear_modulus(CartesianDirection axis,
                                    double angle) const {
  Vec3 shear_dir = shear_direction(axis, angle);
  return shear_modulus(axis, shear_dir);
}

double ElasticTensor::shear_modulus(CartesianDirection a,
                                    CartesianDirection b) const {
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          sum += component(i, j, k, l) * a[i] * b[j] * a[k] * b[l];
        }
      }
    }
  }
  return 0.25 / sum;
}

std::pair<double, double>
ElasticTensor::shear_modulus_minmax(CartesianDirection n) const {
  double min_v = std::numeric_limits<double>::max();
  double max_v = std::numeric_limits<double>::lowest();

  const int num_steps = 360;
  const double step = 2 * M_PI / num_steps;

  for (int i = 0; i < num_steps; ++i) {
    double angle = i * step;
    double v = shear_modulus(n, angle);
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }

  return {min_v, max_v};
}

double ElasticTensor::poisson_ratio_angular(AngularDirection dir,
                                            double angle) const {
  auto [axis, shear_dir] = shear_directions(dir[0], dir[1], angle);
  return poisson_ratio(axis, shear_dir);
}

double ElasticTensor::poisson_ratio(CartesianDirection axis,
                                    double angle) const {
  Vec3 shear_dir = shear_direction(axis, angle);
  return poisson_ratio(axis, shear_dir);
}

double ElasticTensor::poisson_ratio(CartesianDirection a,
                                    CartesianDirection b) const {
  double numerator = 0.0, denominator = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          numerator += m_components[i][j][k][l] * a(i) * a(j) * b(k) * b(l);
          denominator += m_components[i][j][k][l] * a(i) * a(j) * a(k) * a(l);
        }
      }
    }
  }
  return -numerator / denominator;
}

std::pair<double, double>
ElasticTensor::poisson_ratio_minmax(CartesianDirection n) const {
  double min_poisson = std::numeric_limits<double>::max();
  double max_poisson = std::numeric_limits<double>::lowest();

  const int num_steps = 360;
  const double step = 2 * M_PI / num_steps;

  for (int i = 0; i < num_steps; ++i) {
    double angle = i * step;
    double poisson = poisson_ratio(n, angle);
    min_poisson = std::min(min_poisson, poisson);
    max_poisson = std::max(max_poisson, poisson);
  }

  return {min_poisson, max_poisson};
}

double ElasticTensor::average_bulk_modulus(AveragingScheme avg) const {
  double KV = (m_c(0, 0) + m_c(1, 1) + m_c(2, 2) +
               2 * (m_c(0, 1) + m_c(1, 2) + m_c(0, 2))) /
              9.0;
  double KR = 1.0 / (m_s(0, 0) + m_s(1, 1) + m_s(2, 2) +
                     2 * (m_s(0, 1) + m_s(1, 2) + m_s(0, 2)));

  switch (avg) {
  case AveragingScheme::Voigt:
    return KV;
  case AveragingScheme::Reuss:
    return KR;
  case AveragingScheme::Hill:
    return (KV + KR) / 2.0;
  default:
    return (KV + KR) / 2.0; // Default to Hill
  }
}

double ElasticTensor::average_shear_modulus(AveragingScheme avg) const {
  double GV = (m_c(0, 0) + m_c(1, 1) + m_c(2, 2) - m_c(0, 1) - m_c(1, 2) -
               m_c(0, 2) + 3 * (m_c(3, 3) + m_c(4, 4) + m_c(5, 5))) /
              15.0;
  double GR = 15.0 / (4 * (m_s(0, 0) + m_s(1, 1) + m_s(2, 2)) -
                      4 * (m_s(0, 1) + m_s(1, 2) + m_s(0, 2)) +
                      3 * (m_s(3, 3) + m_s(4, 4) + m_s(5, 5)));

  switch (avg) {
  case AveragingScheme::Voigt:
    return GV;
  case AveragingScheme::Reuss:
    return GR;
  case AveragingScheme::Hill:
    return (GV + GR) / 2.0;
  default:
    return (GV + GR) / 2.0; // Default to Hill
  }
}

double ElasticTensor::average_youngs_modulus(AveragingScheme avg) const {
  double K = average_bulk_modulus(avg);
  double G = average_shear_modulus(avg);
  return 9 * K * G / (3 * K + G);
}

double ElasticTensor::average_poisson_ratio(AveragingScheme avg) const {
  double K = average_bulk_modulus(avg);
  double G = average_shear_modulus(avg);
  return (3 * K - 2 * G) / (2 * (3 * K + G));
}

const Mat6 &ElasticTensor::voigt_s() const { return m_s; }

const Mat6 &ElasticTensor::voigt_c() const { return m_c; }

double ElasticTensor::component(int i, int j, int k, int l) const {
  return m_components[i][j][k][l];
}

double &ElasticTensor::component(int i, int j, int k, int l) {
  return m_components[i][j][k][l];
}

} // namespace occ::core
