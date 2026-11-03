#pragma once
#include <occ/core/linear_algebra.h>
#include <cmath>

namespace occ::opt {

struct ConvergenceCriteria {
  // ORCA's NormalOpt: gradients in Hartree/Bohr (Hartree/radian for angles),
  // steps in Bohr (radian), in internal coordinates
  double gradient_max = 3e-4;
  double gradient_rms = 1e-4;
  double step_max = 4e-3;
  double step_rms = 2e-3;

  // Energy change between steps (Hartree), used only on request
  double energy_change = 5e-6;
  bool use_energy_criterion = false;

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
  double trust_radius = 0.3; // Trust radius (s.trust, default value)

  // Control flags
  bool first_step = true; // s.first
  int step_number = 0;
  bool converged = false;

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

} // namespace occ::opt
