/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <fmt/core.h>
#include <cmath>
#include <limits>
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
  double step_norm = step.norm();

  // Fletcher's parameter: ratio of actual to predicted change
  double ratio;
  if (std::abs(predicted_change) < 1e-12) {
    ratio = 1.0; // If predicted change is essentially zero
  } else {
    ratio = actual_change / predicted_change;
  }

  log::debug("Trust update: Fletcher's parameter: {:.6f}", ratio);

  if (ratio < 0.25) {
    return step_norm / 4.0;
  } else if (ratio > 0.75 && std::abs(step_norm - trust) < 1e-10) {
    return 2.0 * trust;
  }

  return trust;
}

inline Vec compute_quadratic_step(const Vec &g, const Mat &H, double trust,
                                  bool &on_sphere, int step_number) {
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

  log::debug("Step {}: {} step, trust={:.6f}, neg_evals={}, min_eval={:.5f}, "
             "lambda={:.6f}",
             step_number, on_sphere ? "sphere-minimised" : "pure RFO", trust,
             (ev.array() < 0).count(), ev(0), lambda);
  log::debug("Step {}: step_rms={:.6f}, step_max={:.6f}, predicted_dE={:.8f}",
             step_number, std::sqrt(dq.squaredNorm() / dq.size()),
             dq.cwiseAbs().maxCoeff(), dE);

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
  state.gradient_cartesian = Mat3N::Zero(3, mol.size());
  state.trust_radius = 0.3;
  state.hessian = coords.hessian_guess();

  // The first point to be evaluated is the input geometry
  state.future.q = coords.to_vector(state.positions);
  state.future.E = 0.0;
  state.future.g = Vec::Zero(coords.size());

  // Best point starts at a very high energy so the first real point replaces
  // it
  state.best.q = Vec::Zero(coords.size());
  state.best.E = std::numeric_limits<double>::max();
  state.best.g = Vec::Zero(coords.size());

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

// One iteration of pyberny's Berny.send(): BFGS and trust-radius updates
// against the best point so far, a linear search between the current and best
// points, then an RFO step from the interpolated point.
bool BernyOptimizer::step() {
  if (state.gradient_cartesian.size() == 0) {
    throw std::runtime_error("BernyOptimizer::step(): No gradient data "
                             "available. Call update() first.");
  }

  // Gradient in internal coordinates, through the generalised inverse of the
  // full (redundant) Wilson B matrix
  Mat B = coords.wilson_b_matrix(state.positions);

  Mat BBT = B * B.transpose();
  Mat BBT_pinv = pseudoinverse(BBT);
  Mat B_inv = B.transpose() * BBT_pinv;

  Vec grad_flat(state.gradient_cartesian.size());
  for (int i = 0; i < state.gradient_cartesian.cols(); i++) {
    grad_flat.segment(3 * i, 3) = state.gradient_cartesian.col(i);
  }

  // The coordinates are those the previous step predicted (future.q), not a
  // fresh evaluation at the new positions
  state.current =
      OptPoint(state.future.q, state.energy, B_inv.transpose() * grad_flat);

  log::debug("Step {}: E={:.8f}, |g|={:.6f}, coords={}({}b,{}a,{}d)",
             state.step_number, state.current.E, state.current.g.norm(),
             coords.size(), coords.bonds().size(), coords.angles().size(),
             coords.dihedrals().size());

  log::debug("Internal coordinate gradients:");
  log::debug("  Type  Atoms              Gradient");
  log::debug("  ----  ----------------   ----------------");
  size_t idx = 0;
  for (size_t i = 0; i < coords.bonds().size(); i++, idx++) {
    const auto &bond = coords.bonds()[i];
    std::string atoms = fmt::format("{}-{}", bond.i, bond.j);
    log::debug("  B     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  for (size_t i = 0; i < coords.angles().size(); i++, idx++) {
    const auto &angle = coords.angles()[i];
    std::string atoms = fmt::format("{}-{}-{}", angle.i, angle.j, angle.k);
    log::debug("  A     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  for (size_t i = 0; i < coords.dihedrals().size(); i++, idx++) {
    const auto &dihedral = coords.dihedrals()[i];
    std::string atoms = fmt::format("{}-{}-{}-{}", dihedral.i, dihedral.j,
                                    dihedral.k, dihedral.l);
    log::debug("  D     {:<16s}   {:12.8f}", atoms, state.current.g(idx));
  }
  log::debug("  Units: B=Hartree/Bohr, A/D=Hartree/radian");

  if (!state.first_step) {
    Vec dq = state.current.q - state.best.q;
    Vec dg = state.current.g - state.best.g;
    if (dq.norm() > 1e-8 && dg.norm() > 1e-8 && dq.dot(dg) > 1e-8) {
      update_hessian_bfgs(state.hessian, dq, dg);
      log::debug("BFGS update applied");
    } else {
      log::debug("BFGS update skipped (insufficient change)");
    }

    // Trust radius from how well the previous step's model predicted the
    // energy change it produced
    double predicted_change = state.predicted.E - state.interpolated.E;
    double actual_change = state.current.E - state.previous.E;
    Vec step_taken = state.predicted.q - state.interpolated.q;
    state.trust_radius = update_trust_radius(
        state.trust_radius, predicted_change, actual_change, step_taken);

    // Linear search between current and best
    Vec dq_interp = state.best.q - state.current.q;
    double g0 = state.current.g.dot(dq_interp);
    double g1 = state.best.g.dot(dq_interp);
    auto [t, E] = linear_search(state.current.E, state.best.E, g0, g1);
    log::debug("Linear search: t={:.6f}, E={:.12f}", t, E);

    state.interpolated.q = state.current.q + t * dq_interp;
    state.interpolated.E = E;
    state.interpolated.g =
        state.current.g + t * (state.best.g - state.current.g);
  } else {
    state.interpolated = state.current;
  }

  if (state.trust_radius < 1e-6) {
    throw std::runtime_error("The trust radius got too small, check forces?");
  }

  // Projected Hessian and gradient: redundant directions are made stiff so
  // the step stays in the space the coordinates can realise
  Mat proj = B * B_inv;
  Mat H_proj = proj * state.hessian * proj +
               1000.0 * (Mat::Identity(coords.size(), coords.size()) - proj);
  Vec g_proj = proj * state.interpolated.g;

  bool on_sphere;
  Vec dq = compute_quadratic_step(g_proj, H_proj, state.trust_radius, on_sphere,
                                  state.step_number);

  state.predicted.q = state.interpolated.q + dq;
  state.predicted.E = state.interpolated.E + state.interpolated.g.dot(dq) +
                      0.5 * dq.transpose() * state.hessian * dq;
  state.predicted.g = Vec::Zero(coords.size()); // Not used

  log::debug("Predicted energy change: {:.8f}",
             state.predicted.E - state.current.E);
  if (dq.norm() > state.trust_radius * 1.01) {
    log::error("BUG: Internal step exceeds trust radius!");
  }

  Vec current_to_predicted = state.predicted.q - state.current.q;

  // Converged at the current (evaluated) geometry: its gradient and the step
  // the model proposes from it are both small. The geometry is then left
  // where it is, so the result is a point whose energy was computed.
  const double energy_change =
      state.first_step ? 0.0 : state.current.E - state.previous.E;
  auto conv_info = check_convergence(state.current.g, current_to_predicted,
                                     energy_change, state.criteria);
  log::info("Step {:2d} convergence check (internal coordinates):",
            state.step_number);
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
  if (state.criteria.use_energy_criterion && !state.first_step)
    log::info("  Energy change:   {:.5e} (threshold: {:.2e}) {}",
              conv_info.energy_change, state.criteria.energy_change,
              conv_info.energy_converged ? "✓" : "✗");
  if (conv_info.converged) {
    state.converged = true;
    log::info("Optimization converged!");
    return true;
  }

  // Take the step from the current point (pyberny's update_geom): the
  // back-transformation may not reach predicted.q exactly, so the next
  // point's coordinates are those actually reached
  auto [new_q, new_positions] = update_geometry(
      state.current.q, current_to_predicted, coords, state.positions, B_inv);
  state.positions = new_positions;
  state.future = OptPoint(new_q, 0.0, Vec::Zero(new_q.size()));

  state.previous = state.current;
  if (state.first_step || state.current.E < state.best.E) {
    state.best = state.current;
    log::debug("Updated best point to current: E={:.8f}", state.best.E);
  } else {
    log::debug("Kept previous best point: E={:.8f}", state.best.E);
  }

  state.first_step = false;
  state.step_number++;
  return false;
}

core::Molecule BernyOptimizer::get_next_geometry() const {
  return core::Molecule(molecule.atomic_numbers(), state.positions);
}

void BernyOptimizer::shift_energy_reference(double shift) {
  state.energy += shift;
  state.current.E += shift;
  state.previous.E += shift;
  if (state.best.E != std::numeric_limits<double>::max())
    state.best.E += shift;
  state.future.E += shift;
}

void BernyOptimizer::update(double energy, const Mat3N &gradient) {
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
}

} // namespace occ::opt
