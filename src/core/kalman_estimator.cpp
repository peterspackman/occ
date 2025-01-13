#include <occ/core/kalman_estimator.h>

namespace occ::core {

void KalmanEstimator::update(double measurement, double now) {
  if (!initialized) {
    x[0] = measurement;
    last_update = now;
    initialized = true;
    return;
  }

  double dt = now - last_update;

  if (dt > 0) {
    Mat2 F;
    F << 1, dt, 0, 1;

    x = F * x;
    P = F * P * F.transpose() + Q;

    // Update step
    double H_0 = 1.0; // Measurement model [1, 0]
    double S = (H_0 * P(0, 0) * H_0 + R);
    Vec2 K(P(0, 0) / S, P(1, 0) / S); // Kalman gain

    double innovation = measurement - x[0];
    x += K * innovation;
    P = (Mat2::Identity() - K * Vec2(H_0, 0).transpose()) * P;

    last_update = now;
  }
}

double KalmanEstimator::estimate_remaining(double total) const {
  if (!initialized || x[1] <= 0) {
    return 0.0;
  }

  double remaining = total - x[0];
  return remaining / x[1];
}

void KalmanEstimator::adjust_noise(double process_noise,
                                   double measurement_noise) {
  Q = Mat2::Identity() * process_noise;
  R = measurement_noise;
}

double KalmanEstimator::time_uncertainty(double total) const {
  if (!initialized || x[1] <= 0)
    return 0.0;

  double remaining = total - x[0];
  // Propagate uncertainty using first-order approximation
  double pos_uncert = P(0, 0);
  double vel_uncert = P(1, 1);
  double pos_vel_cov = P(0, 1);

  // Variance of t = remaining/velocity using error propagation
  double variance =
      (remaining * remaining * vel_uncert / (x[1] * x[1] * x[1] * x[1])) +
      (pos_uncert / (x[1] * x[1])) -
      (2 * remaining * pos_vel_cov / (x[1] * x[1] * x[1]));

  return std::sqrt(std::max(0.0, variance));
}

} // namespace occ::core
