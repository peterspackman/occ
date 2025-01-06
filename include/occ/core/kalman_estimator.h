#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {
struct KalmanEstimator {

  Vec2 x{Vec2::Zero()};              // State: [position, velocity]
  Mat2 P{Mat2::Identity() * 1000.0}; // Covariance
  Mat2 Q{Mat2::Identity() * 0.1};    // Process noise
  double R{1.0};                     // Measurement noise

  double last_update{0.0};
  bool initialized{false};

  void update(double measurement, double now);
  double estimate_remaining(double total) const;
  void adjust_noise(double process_noise, double measurement_noise);
  inline const auto &state() const { return x; }
  inline const auto &covariance() const { return P; }
  double time_uncertainty(double total) const;
};

} // namespace occ::core
