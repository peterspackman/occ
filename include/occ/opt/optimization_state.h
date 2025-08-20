#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::opt {

struct ConvergenceCriteria {
  // Gradient convergence thresholds (Hartree/Angstrom for Cartesian, unitless for internal coords)
  double gradient_max = 0.45e-3; // Maximum gradient component 
  double gradient_rms = 0.15e-3; // Root-mean-square of gradient
  
  // Step convergence thresholds (Angstrom for Cartesian, unitless for internal coords)
  double step_max = 1.8e-3;      // Maximum displacement component
  double step_rms = 1.2e-3;      // Root-mean-square of displacement
  
  // Energy convergence (optional, in Hartree)
  double energy_change = 1e-6;   // Maximum energy change between steps
  bool use_energy_criterion = false; // Whether to use energy as a convergence criterion
  
  // Maximum iterations
  int max_iterations = 100;
};

// Optimization point structure
struct OptPoint {
  Vec q;    // Internal coordinates
  double E; // Energy
  Vec g;    // Gradient in internal coordinates

  OptPoint() : E(0.0) {}
  OptPoint(const Vec &coords, double energy, const occ::Vec &grad)
      : q(coords), E(energy), g(grad) {}
};

// Optimizer state - matches Berny State class
struct OptimizationState {
  // Geometry and coordinates
  Mat3N positions; // Current Cartesian positions (Angstroms)
  Vec current_q;   // Current internal coordinates

  // Energy and gradients from latest calculation
  double energy = 0.0;
  Mat3N gradient_cartesian; // Gradient in Cartesian coordinates

  // Berny state points
  OptPoint current;      // Current point from energy/gradient evaluation
  OptPoint best;         // Best point seen so far
  OptPoint previous;     // Previous point
  OptPoint interpolated; // Interpolated between current and best
  OptPoint predicted;    // Predicted next point
  OptPoint future;       // Future point (for next iteration)

  // Optimization data
  Mat hessian; // Approximate Hessian in internal coordinates (s.H)
  Vec weights; // Coordinate weights (s.weights)
  double trust_radius = 0.3; // Trust radius (s.trust, default value)

  // Control flags
  bool first_step = true; // s.first
  int step_number = 0;
  bool converged = false;

  // History for debugging/analysis
  std::vector<OptPoint> history;

  ConvergenceCriteria criteria;
};

struct ConvergenceInfo {
  double max_gradient;
  double rms_gradient;
  double max_step;
  double rms_step;
  double energy_change;
  
  // Individual criteria status
  bool gradient_max_converged;
  bool gradient_rms_converged;
  bool step_max_converged;
  bool step_rms_converged;
  bool energy_converged;
  
  // Overall convergence
  bool converged;
};

inline ConvergenceInfo check_convergence(const Vec &gradient, const Vec &step,
                                         double energy_change,
                                         const ConvergenceCriteria &criteria) {
  ConvergenceInfo info;
  
  // Calculate values
  info.max_gradient = gradient.array().abs().maxCoeff();
  info.rms_gradient = std::sqrt(gradient.squaredNorm() / gradient.size());
  info.max_step = step.array().abs().maxCoeff();
  info.rms_step = std::sqrt(step.squaredNorm() / step.size());
  info.energy_change = std::abs(energy_change);
  
  // Check individual criteria
  info.gradient_max_converged = info.max_gradient < criteria.gradient_max;
  info.gradient_rms_converged = info.rms_gradient < criteria.gradient_rms;
  info.step_max_converged = info.max_step < criteria.step_max;
  info.step_rms_converged = info.rms_step < criteria.step_rms;
  info.energy_converged = !criteria.use_energy_criterion || 
                          (info.energy_change < criteria.energy_change);

  // Overall convergence requires all active criteria to be met
  info.converged = info.gradient_max_converged &&
                   info.gradient_rms_converged &&
                   info.step_max_converged &&
                   info.step_rms_converged &&
                   info.energy_converged;

  return info;
}

// Backward compatibility overload
inline ConvergenceInfo check_convergence(const Vec &gradient, const Vec &step,
                                         const ConvergenceCriteria &criteria) {
  return check_convergence(gradient, step, 0.0, criteria);
}

} // namespace occ::opt
