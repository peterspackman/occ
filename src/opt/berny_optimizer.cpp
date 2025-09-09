/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <cmath>
#include <occ/core/log.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/linear_search.h>
#include <occ/opt/pseudoinverse.h>

namespace occ::opt {

// Find root of increasing function on (-inf, lim) using Newton-Raphson
// Assumes f(-inf) < 0, f(lim) > 0
template <typename Func> inline double find_root_newton(Func f, double lim) {
  double d = 1.0;

  // Find d so that f(lim-d) > 0
  for (int i = 0; i < 1000; i++) {
    double val = f(lim - d);
    if (val > 0)
      break;
    d = d / 2.0;
  }

  double x = lim - d; // initial guess
  double dx = 1e-10;  // step for numerical derivative
  double fx = f(x);
  double err = std::abs(fx);

  // Newton-Raphson iterations
  for (int i = 0; i < 1000; i++) {
    double fxpdx = f(x + dx);
    double dxf = (fxpdx - fx) / dx;
    x = x - fx / dxf;
    fx = f(x);
    double err_new = std::abs(fx);
    if (err_new >= err)
      break;
    err = err_new;
    if (err < 1e-10)
      break; // Converged
  }

  return x;
}

// Forward declarations for internal helper functions
inline void update_hessian_bfgs(Mat &H, const Vec &dq, const Vec &dg) {
  double dq_dg = dq.dot(dg);

  if (dq_dg < 1e-8) {
    return; // Skip update if vectors are nearly orthogonal
  }

  Vec Hdq = H * dq;
  double dq_H_dq = dq.dot(Hdq);

  // BFGS update formula
  H += (dg * dg.transpose()) / dq_dg - (Hdq * Hdq.transpose()) / dq_H_dq;
}

inline double update_trust_radius(double trust, double predicted_change,
                                  double actual_change, const Vec &step) {
  // Trust radius update function
  double step_norm = step.norm();

  // Fletcher's parameter: ratio of actual to predicted change
  double ratio;
  if (std::abs(predicted_change) < 1e-12) {
    ratio = 1.0; // If predicted change is essentially zero
  } else {
    ratio = actual_change / predicted_change;
  }

  log::debug("Trust update: Fletcher's parameter: {:.6f}", ratio);

  // Update trust radius based on Fletcher's parameter
  if (ratio < 0.25) {
    return step_norm / 4.0;
  } else if (ratio > 0.75 && std::abs(step_norm - trust) < 1e-10) {
    return 2.0 * trust;
  }

  return trust; // Keep current trust radius
}

inline Vec compute_quadratic_step(const Vec &g, const Mat &H, double trust,
                                  bool &on_sphere, int step_number = -1) {
  // Compute eigenvalues of symmetrized Hessian
  Mat H_sym = (H + H.transpose()) / 2.0;
  Eigen::SelfAdjointEigenSolver<Mat> eigen_solver(H_sym);
  Vec ev = eigen_solver.eigenvalues();

  // Rational Function Optimization (RFO) approach
  int n = H.rows();
  Mat rfo(n + 1, n + 1);
  rfo.block(0, 0, n, n) = H;
  rfo.block(0, n, n, 1) = g;
  rfo.block(n, 0, 1, n) = g.transpose();
  rfo(n, n) = 0.0;

  // Symmetrize and solve eigenvalue problem
  Mat rfo_sym = (rfo + rfo.transpose()) / 2.0;
  Eigen::SelfAdjointEigenSolver<Mat> rfo_solver(rfo_sym);
  Mat V = rfo_solver.eigenvectors();
  Vec D = rfo_solver.eigenvalues();

  // Extract step from eigenvector: dq = V[:-1, 0] / V[-1, 0], lambda = D[0]
  Vec dq = V.block(0, 0, n, 1) / V(n, 0);
  double lambda = D(0);

  log::trace("RFO: g={:.6f}, H_diag={:.6f}, raw_dq={:.6f}, lambda={:.6f}", g(0),
             H(0, 0), dq(0), lambda);

  // If step is within trust radius, use it directly
  if (dq.norm() <= trust) {
    on_sphere = false;
  } else {
    // Minimize on trust radius sphere
    auto steplength = [&](double l) -> double {
      Mat lI_minus_H = l * Mat::Identity(n, n) - H;
      Eigen::LDLT<Mat> ldlt(lI_minus_H);
      if (ldlt.info() != Eigen::Success) {
        return 1e6; // Large value if singular
      }
      Vec step = ldlt.solve(g);
      return step.norm() - trust;
    };

    // Find root to constrain step to trust radius
    lambda = find_root_newton(steplength, ev(0));

    // Solve constrained step: dq = inv(lambda * I - H) * g
    Mat lI_minus_H = lambda * Mat::Identity(n, n) - H;
    Eigen::LDLT<Mat> ldlt(lI_minus_H);
    dq = ldlt.solve(g);
    on_sphere = true;
  }

  // Predicted energy change from quadratic model
  double dE = g.dot(dq) + 0.5 * dq.transpose() * H * dq;

  // RFO step logging
  if (step_number >= 0) {
    double step_rms = std::sqrt(dq.squaredNorm() / dq.size());
    double step_max = dq.cwiseAbs().maxCoeff();

    // Step summary for debug level
    log::debug("Step {}: trust={:.3f}, step_rms={:.4f}, dE={:.5f}", step_number,
               trust, step_rms, dE);

    // Detailed step info for debug level
    if (on_sphere) {
      log::debug("Step {}: sphere minimization performed", step_number);
    } else {
      log::debug("Step {}: pure RFO step performed", step_number);
    }
    log::debug("Step {}: trust_radius={:.6f}, neg_evals={}, "
               "min_eval={:.5f}, lambda={:.6f}",
               step_number, trust, (ev.array() < 0).count(), ev(0), lambda);
    log::debug("Step {}: step_rms={:.6f}, step_max={:.6f}, predicted_dE={:.8f}",
               step_number, step_rms, step_max, dE);
  }

  return dq;
}

BernyOptimizer::BernyOptimizer(const core::Molecule &mol,
                               const ConvergenceCriteria &criteria)
    : molecule(mol), coords(mol, {true}) {
  state.positions = mol.positions();
  state.criteria = criteria;
  state.step_number = 0;
  state.converged = false;
  state.energy = 0.0;
  state.first_step = true;

  // Initialize gradient to zero
  state.gradient_cartesian = Mat3N::Zero(3, mol.size());

  // Use default trust radius
  state.trust_radius = 0.3;

  // (gradient_internal removed - now computed on demand)

  // Initialize Hessian guess
  state.hessian = coords.hessian_guess();

  // Initialize current internal coordinates
  state.current_q = coords.to_vector(state.positions);

  // Initialize optimization state points
  state.future.q = state.current_q;
  state.future.E = 0.0; // Will be set when we get first energy
  state.future.g = Vec::Zero(coords.size());

  // Initialize best point to a very high energy so first real point becomes
  // best
  state.best.q = Vec::Zero(coords.size());
  state.best.E = std::numeric_limits<double>::max();
  state.best.g = Vec::Zero(coords.size());

  // Initialize weights
  state.weights = Vec::Ones(coords.size()); // Simple uniform weights for now

  // Validate initialization
  if (coords.size() == 0) {
    throw std::runtime_error(
        "BernyOptimizer: No internal coordinates generated");
  }
  if (state.hessian.rows() != static_cast<int>(coords.size())) {
    throw std::runtime_error(
        "BernyOptimizer: Hessian size mismatch with coordinates");
  }

  log::info("BernyOptimizer initialized with {} internal coordinates",
            coords.size());
  log::debug("  {} bonds, {} angles, {} dihedrals", coords.bonds().size(),
             coords.angles().size(), coords.dihedrals().size());
}

bool BernyOptimizer::step() {

  // This follows Berny algorithm's send() method
  log::trace("===== STEP {} PROCESSING START =====", state.step_number);
  log::trace("Input energy: {:.12f}", state.energy);

  // Log state at BEGINNING of step
  log::trace("STATE AT STEP START:");
  if (state.predicted.q.size() > 0) {
    log::trace("  predicted.E: {:.12f}", state.predicted.E);
    log::trace("  predicted.q: [{:.12f}]", state.predicted.q(0));
  } else {
    log::trace("  predicted: EMPTY");
  }
  if (state.interpolated.q.size() > 0) {
    log::trace("  interpolated.E: {:.12f}", state.interpolated.E);
    log::trace("  interpolated.q: [{:.12f}]", state.interpolated.q(0));
  } else {
    log::trace("  interpolated: EMPTY");
  }
  if (state.previous.q.size() > 0) {
    log::trace("  previous.E: {:.12f}", state.previous.E);
    log::trace("  previous.q: [{:.12f}]", state.previous.q(0));
  } else {
    log::trace("  previous: EMPTY");
  }
  if (state.best.q.size() > 0) {
    log::trace("  best.E: {:.12f}", state.best.E);
    log::trace("  best.q: [{:.12f}]", state.best.q(0));
  } else {
    log::trace("  best: EMPTY");
  }

  // Validate that we have gradient data
  if (state.gradient_cartesian.size() == 0) {
    throw std::runtime_error("BernyOptimizer::step(): No gradient data "
                             "available. Call update() first.");
  }

  // Compute B-matrix and transform gradient to internal coordinates
  // Use FULL B-matrix - redundancy handled in pseudoinverse
  Mat B = coords.wilson_b_matrix(state.positions);

  Mat BBT = B * B.transpose();
  Mat BBT_pinv = pseudoinverse(BBT);
  Mat B_inv = B.transpose() * BBT_pinv;

  Vec grad_flat(state.gradient_cartesian.size());
  for (int i = 0; i < state.gradient_cartesian.cols(); i++) {
    grad_flat.segment(3 * i, 3) = state.gradient_cartesian.col(i);
  }

  // Create current optimization point
  // Use s.future.q from previous step, NOT fresh computation
  state.current =
      OptPoint(state.future.q, state.energy, B_inv.transpose() * grad_flat);

  // Energy info for debug level
  log::debug("Step {}: E={:.8f}", state.step_number, state.current.E);

  // Detailed coordinate summary for debug
  log::debug("Step {}: coords={}({}b,{}a,{}d)", state.step_number,
             coords.size(), coords.bonds().size(), coords.angles().size(),
             coords.dihedrals().size());

  log::debug("Step {}: E={:.8f}, |g|={:.6f}", state.step_number,
             state.current.E, state.current.g.norm());
  
  // Log internal coordinate gradients at debug level
  log::debug("Internal coordinate gradients:");
  log::debug("  Type  Atoms              Gradient");
  log::debug("  ----  ----------------   ----------------");
  size_t idx = 0;
  // Bonds (unitless - derivative of energy w.r.t. bond length in Angstrom)
  for (size_t i = 0; i < coords.bonds().size(); i++, idx++) {
    const auto &bond = coords.bonds()[i];
    std::string atoms = fmt::format("{}-{}", bond.i, bond.j);
    log::debug("  B     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  // Angles (units: Hartree/radian)
  for (size_t i = 0; i < coords.angles().size(); i++, idx++) {
    const auto &angle = coords.angles()[i];
    std::string atoms = fmt::format("{}-{}-{}", angle.i, angle.j, angle.k);
    log::debug("  A     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  // Dihedrals (units: Hartree/radian)
  for (size_t i = 0; i < coords.dihedrals().size(); i++, idx++) {
    const auto &dihedral = coords.dihedrals()[i];
    std::string atoms = fmt::format("{}-{}-{}-{}", 
                                    dihedral.i, dihedral.j, dihedral.k, dihedral.l);
    log::debug("  D     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  if (coords.bonds().size() > 0) {
    log::debug("  Units: B=Hartree/Angstrom, A/D=Hartree/radian");
  }
  log::trace(
      "GEOMETRY STEP {}: [{:.8f}, {:.8f}, {:.8f}] [{:.8f}, {:.8f}, {:.8f}]",
      state.step_number, state.positions(0, 0), state.positions(1, 0),
      state.positions(2, 0), state.positions(0, 1), state.positions(1, 1),
      state.positions(2, 1));

  if (!state.first_step) {
    log::trace("Processing subsequent step - doing BFGS and trust updates");

    // Update Hessian with BFGS
    Vec dq = state.current.q - state.best.q;
    Vec dg = state.current.g - state.best.g;

    log::trace("BFGS update: dq norm = {:.6f}, dg norm = {:.6f}", dq.norm(),
               dg.norm());

    if (dq.norm() > 1e-8 && dg.norm() > 1e-8 && dq.dot(dg) > 1e-8) {
      update_hessian_bfgs(state.hessian, dq, dg);
      log::debug("BFGS update applied");
    } else {
      log::debug("BFGS update skipped (insufficient change)");
    }

    // Trust radius update
    // Use previous step's predicted and interpolated values (from previous
    // iteration)
    double predicted_change = state.predicted.E - state.interpolated.E;
    double actual_change = state.current.E - state.previous.E;
    Vec step_taken = state.predicted.q - state.interpolated.q;

    log::trace("Trust update:");
    log::trace("  current.E: {:.12f}", state.current.E);
    log::trace("  previous.E: {:.12f}", state.previous.E);
    log::trace("  actual_change: {:.12f}", actual_change);
    log::trace("  predicted.E: {:.12f}", state.predicted.E);
    log::trace("  interpolated.E: {:.12f}", state.interpolated.E);
    log::trace("  predicted_change: {:.12f}", predicted_change);
    log::trace("  step_taken norm: {:.12f}", step_taken.norm());

    state.trust_radius = update_trust_radius(
        state.trust_radius, predicted_change, actual_change, step_taken);

    // Linear interpolation between current and best
    Vec dq_interp = state.best.q - state.current.q;
    double g0 = state.current.g.dot(dq_interp);
    double g1 = state.best.g.dot(dq_interp);

    log::trace("Linear search setup:");
    log::trace("  dq (best - current): {:.6f}", dq_interp(0));
    log::trace("  current.g: {:.6f}", state.current.g(0));
    log::trace("  best.g: {:.6f}", state.best.g(0));
    log::trace("  current.g · dq: {:.6f}", g0);
    log::trace("  best.g · dq: {:.6f}", g1);

    // Use linear search method
    auto [t, E] = linear_search(state.current.E, state.best.E, g0, g1);

    log::debug("Linear search: t={:.6f}, E={:.12f}", t, E);

    // Update interpolated point for this iteration
    state.interpolated.q = state.current.q + t * dq_interp;
    state.interpolated.E = E; // Use the energy from linear fit
    state.interpolated.g =
        state.current.g + t * (state.best.g - state.current.g);

  } else {
    // First step: set interpolated = current
    state.interpolated = state.current;
    log::debug("First step - using current as interpolated");
  }

  // Check convergence before computing new step
  double grad_rms = std::sqrt(state.interpolated.g.dot(state.interpolated.g) /
                              state.interpolated.g.size());
  double grad_max = state.interpolated.g.cwiseAbs().maxCoeff();

  // Will check step convergence after we compute the step
  // For now just log current gradient status
  log::trace("Pre-step convergence: RMS grad={:.6f} (tol={:.6f}), Max "
             "grad={:.6f} (tol={:.6f})",
             grad_rms, state.criteria.gradient_rms, grad_max,
             state.criteria.gradient_max);

  if (grad_rms < state.criteria.gradient_rms &&
      grad_max < state.criteria.gradient_max) {
    state.converged = true;
    log::info("Optimization converged!");
    return true;
  }

  // Check trust radius
  if (state.trust_radius < 1e-6) {
    throw std::runtime_error("The trust radius got too small, check forces?");
  }

  // Compute projected Hessian and gradient
  Mat proj = B * B_inv;
  Mat H_proj = proj * state.hessian * proj +
               1000.0 * (Mat::Identity(coords.size(), coords.size()) - proj);
  Vec g_proj = proj * state.interpolated.g;

  // Quadratic step with trust region
  bool on_sphere;
  Vec dq = compute_quadratic_step(g_proj, H_proj, state.trust_radius, on_sphere,
                                  state.step_number);

  // Predicted point
  state.predicted.q = state.interpolated.q + dq;
  state.predicted.E = state.interpolated.E + state.interpolated.g.dot(dq) +
                      0.5 * dq.transpose() * state.hessian * dq;
  state.predicted.g = Vec::Zero(coords.size()); // Not used

  log::debug("Trust region step norm: {:.6f}", dq.norm());
  log::debug("Predicted energy change: {:.8f}",
             state.predicted.E - state.current.E);

  // Check if trust radius is actually being enforced
  log::trace("Step size check: trust={:.8f}, step_norm={:.8f}, ratio={:.2f}",
             state.trust_radius, dq.norm(), dq.norm() / state.trust_radius);
  if (dq.norm() > state.trust_radius * 1.01) {
    log::error("BUG: Internal step exceeds trust radius!");
  }

  // Compute the step from current to predicted state
  Vec current_to_predicted = state.predicted.q - state.current.q;

  double step_rms = std::sqrt(current_to_predicted.squaredNorm() /
                              current_to_predicted.size());
  double step_norm = current_to_predicted.norm();
  double step_max = current_to_predicted.cwiseAbs().maxCoeff();

  log::debug("Step {}: step_rms={:.4f}, step_max={:.3f}, step_norm={:.6f}",
             state.step_number, step_rms, step_max, step_norm);
  log::debug("{} Step: current->predicted norm={:.6f}", state.step_number,
             step_norm);

  log::trace("dq {}", format_matrix(current_to_predicted));
  log::trace("Internal step: dq = [{:.8f}]", current_to_predicted(0));
  log::trace("Current q: [{:.8f}], predicted q: [{:.8f}]", state.current.q(0),
             state.predicted.q(0));

  // Use update_geometry logic: q, s.geom = s.coords.update_geom(...)
  // Use the step from current to predicted (like s.future.q - current.q)
  auto [new_q, new_positions] = update_geometry(
      state.current.q, current_to_predicted, coords, state.positions, B_inv);

  // Log transformation results
  // The convergence info is logged inside update_geometry function

  log::trace("Before position update: [{:.8f}, {:.8f}]", state.positions(0, 1),
             state.positions(0, 0));
  Mat3N cart_step = new_positions - state.positions;
  log::trace("Cartesian step: [{:.8f}, {:.8f}]", cart_step(0, 0),
             cart_step(0, 1));
  state.positions = new_positions;

  // Set future = OptPoint(q, None, None) - use the NEW q from update_geometry
  state.future = OptPoint(
      new_q, 0.0,
      Vec::Zero(new_q.size())); // Will be updated with real energy/gradient
  log::trace("After position update: [{:.8f}, {:.8f}]", state.positions(0, 1),
             state.positions(0, 0));

  // Check final bond length
  double final_bond = (state.positions.col(0) - state.positions.col(1)).norm();
  log::trace("Final bond length: {:.8f}, expected: {:.8f}", final_bond,
             state.predicted.q(0));

  // Set future and previous optimization points
  // future.q already set correctly above from update_geometry
  // Just reset energy and gradient for next iteration
  state.future.E = 0.0; // Will be set when we get next energy
  state.future.g = Vec::Zero(state.future.q.size()); // Match the q size

  state.previous = state.current;

  // Update best point at the end
  if (state.first_step || state.current.E < state.best.E) {
    state.best = state.current;
    log::debug("Updated best point to current: E={:.8f}", state.best.E);
  } else {
    log::debug("Kept previous best point: E={:.8f}", state.best.E);
  }

  // Convergence check using unified system
  double energy_change = state.first_step ? 0.0 : 
                         (state.current.E - state.previous.E);
  
  auto conv_info = check_convergence(state.current.g, current_to_predicted,
                                     energy_change, state.criteria);
  
  // Clear convergence status log
  log::info("Step {:2d} convergence check (internal coordinates):", state.step_number);
  log::info("  RMS gradient:    {:.5e} (threshold: {:.2e}) {}",
            conv_info.rms_gradient, state.criteria.gradient_rms,
            conv_info.gradient_rms_converged ? "✓" : "✗");
  log::info("  Max gradient:    {:.5e} (threshold: {:.2e}) {}",
            conv_info.max_gradient, state.criteria.gradient_max,
            conv_info.gradient_max_converged ? "✓" : "✗");
  log::info("  RMS step:        {:.5e} (threshold: {:.2e}) {}",
            conv_info.rms_step, state.criteria.step_rms,
            conv_info.step_rms_converged ? "✓" : "✗");
  log::info("  Max step:        {:.5e} (threshold: {:.2e}) {}",
            conv_info.max_step, state.criteria.step_max,
            conv_info.step_max_converged ? "✓" : "✗");
  
  if (state.criteria.use_energy_criterion && !state.first_step) {
    log::info("  Energy change:   {:.5e} (threshold: {:.2e}) {}",
              conv_info.energy_change, state.criteria.energy_change,
              conv_info.energy_converged ? "✓" : "✗");
  }
  
  state.converged = conv_info.converged;

  state.first_step = false;
  state.step_number++;

  // Log state at END of step
  log::trace("STATE AT STEP END:");
  log::trace("  predicted.E: {:.12f}", state.predicted.E);
  log::trace("  predicted.q: [{:.12f}]", state.predicted.q(0));
  log::trace("  interpolated.E: {:.12f}", state.interpolated.E);
  log::trace("  interpolated.q: [{:.12f}]", state.interpolated.q(0));
  log::trace("  previous.E: {:.12f}", state.previous.E);
  log::trace("  previous.q: [{:.12f}]", state.previous.q(0));
  log::debug("===== STEP {} PROCESSING END =====", state.step_number - 1);

  return false;
}

core::Molecule BernyOptimizer::get_next_geometry() const {
  return core::Molecule(molecule.atomic_numbers(), state.positions);
}

void BernyOptimizer::update(double energy, const Mat3N &gradient) {
  // Validate gradient dimensions
  if (gradient.rows() != 3) {
    throw std::runtime_error(
        "BernyOptimizer::update(): Gradient must have 3 rows (x,y,z)");
  }
  if (gradient.cols() != molecule.size()) {
    throw std::runtime_error(
        "BernyOptimizer::update(): Gradient size doesn't match molecule size");
  }

  state.energy = energy;
  state.gradient_cartesian = gradient;

  log::trace("Updated with energy = {:.12f}", energy);
}

} // namespace occ::opt
