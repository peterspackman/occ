#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/opt/lbfgs_optimizer.h>
#include <stdexcept>

namespace occ::opt {

LbfgsOptimizer::LbfgsOptimizer(const core::Molecule &mol,
                               const ConvergenceCriteria &criteria,
                               const LbfgsSettings &settings)
    : m_molecule(mol), m_criteria(criteria), m_settings(settings) {
  if (settings.memory < 1)
    throw std::invalid_argument("LbfgsOptimizer: memory must be >= 1");
  if (!(settings.min_step > 0.0 && settings.min_step <= settings.max_step))
    throw std::invalid_argument(
        "LbfgsOptimizer: need 0 < min_step <= max_step");
  m_trust = settings.max_step;
  const Mat3N bohr = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
  m_x = bohr.reshaped();
  const int n = m_x.size();
  m_s = Mat::Zero(n, settings.memory);
  m_y = Mat::Zero(n, settings.memory);
  m_rho = Vec::Zero(settings.memory);
  if (settings.precondition) {
    m_coords = std::make_unique<InternalCoordinates>(
        mol, InternalCoordinates::Options{true, false});
    m_model_hessian = m_coords->hessian_guess();
  }
}

void LbfgsOptimizer::build_preconditioner() {
  const Mat3N angstrom =
      m_x.reshaped(3, m_x.size() / 3) * occ::units::BOHR_TO_ANGSTROM;
  const Mat B = m_coords->wilson_b_matrix(angstrom);
  Mat P = B.transpose() * m_model_hessian * B;
  P.diagonal().array() += m_settings.precondition_shift;
  m_precondition.compute(P);
}

Vec LbfgsOptimizer::apply_h0(const Vec &q) const {
  if (m_coords)
    return m_precondition.solve(q);
  return q / m_settings.initial_hessian;
}

void LbfgsOptimizer::update(double energy, const Mat3N &gradient) {
  if (gradient.rows() != 3 || gradient.cols() != m_molecule.size())
    throw std::runtime_error(
        "LbfgsOptimizer::update(): gradient must be 3 x N atoms");
  m_energy = energy;
  m_g = gradient.reshaped();
}

void LbfgsOptimizer::shift_energy_reference(double shift) {
  m_energy += shift;
  m_energy_prev += shift;
}

void LbfgsOptimizer::push_pair(const Vec &s, const Vec &y) {
  const double ys = y.dot(s);
  if (ys <= 1e-15) {
    log::debug("L-BFGS: skipping pair with y.s = {:.3e}", ys);
    return;
  }
  m_s.col(m_cursor) = s;
  m_y.col(m_cursor) = y;
  m_rho(m_cursor) = 1.0 / ys;
  m_cursor = (m_cursor + 1) % m_settings.memory;
  m_num_stored = std::min(m_num_stored + 1, m_settings.memory);
}

// Two-loop recursion: -H_k g, oldest pair at (cursor - num_stored).
Vec LbfgsOptimizer::direction(const Vec &g) const {
  const int m = m_settings.memory;
  if (m_num_stored == 0)
    return -apply_h0(g);
  auto slot = [&](int i) { return (m_cursor - m_num_stored + i + m) % m; };

  Vec q = g;
  Vec alpha(m_num_stored);
  for (int i = m_num_stored - 1; i >= 0; --i) {
    const int k = slot(i);
    alpha(i) = m_rho(k) * m_s.col(k).dot(q);
    q -= alpha(i) * m_y.col(k);
  }
  if (m_coords) {
    q = m_precondition.solve(q);
  } else {
    const int last = slot(m_num_stored - 1);
    q *= m_s.col(last).dot(m_y.col(last)) / m_y.col(last).squaredNorm();
  }
  for (int i = 0; i < m_num_stored; ++i) {
    const int k = slot(i);
    const double beta = m_rho(k) * m_y.col(k).dot(q);
    q += (alpha(i) - beta) * m_s.col(k);
  }
  return -q;
}

bool LbfgsOptimizer::step() {
  if (m_g.size() != m_x.size())
    throw std::runtime_error("LbfgsOptimizer::step() called before update()");

  bool rejected = false;
  if (m_x_prev.size() > 0) {
    // The gradient at the new point is information either way.
    push_pair(m_x - m_x_prev, m_g - m_g_prev);
    const double actual = m_energy - m_energy_prev;
    const double ratio =
        m_predicted_change < 0.0 ? actual / m_predicted_change : 1.0;
    if (ratio < 0.25)
      m_trust = std::max(m_settings.min_step, 0.5 * m_last_disp);
    else if (ratio > 0.75 && m_last_disp > 0.99 * m_trust)
      m_trust = std::min(m_settings.max_step, 2.0 * m_trust);
    log::debug("L-BFGS: dE {:.3e}, predicted {:.3e}, ratio {:.3f}, trust {:.3e}",
               actual, m_predicted_change, ratio, m_trust);
    if (m_settings.reject_uphill && actual > 0.0) {
      log::info("L-BFGS: energy rose by {:.3e} Ha, retaking the step from the "
                "previous point (trust radius {:.3e} Bohr)",
                actual, m_trust);
      m_x = m_x_prev;
      m_g = m_g_prev;
      m_energy = m_energy_prev;
      rejected = true;
    }
  }

  if (m_coords)
    build_preconditioner();
  Vec d = direction(m_g);
  if (d.dot(m_g) > 0) {
    log::info("L-BFGS: uphill direction, clearing history");
    m_num_stored = 0;
    m_cursor = 0;
    d = -apply_h0(m_g);
  }

  // Scale to the trust radius. The quadratic model along d = -H g, with
  // B = H^-1, predicts dE(a d) = a g.d - a^2 g.d / 2.
  const double gd = m_g.dot(d);
  double max_disp = d.reshaped(3, d.size() / 3).colwise().norm().maxCoeff();
  double scale = 1.0;
  if (max_disp > m_trust) {
    scale = m_trust / max_disp;
    d *= scale;
    max_disp = m_trust;
  }

  const double energy_change = m_step == 0 ? 0.0 : m_energy - m_energy_prev;
  auto info = check_convergence(m_g, d, energy_change, m_criteria);
  log::info("Step {:2d} convergence check (Cartesian, L-BFGS, {} pairs):",
            m_step, m_num_stored);
  log::info("  RMS gradient:    {:.5e} (threshold: {:.2e}) {}",
            info.rms_gradient, m_criteria.gradient_rms,
            info.gradient_rms_converged ? "✓" : "✗");
  log::info("  Max gradient:    {:.5e} (threshold: {:.2e}) {}",
            info.max_gradient, m_criteria.gradient_max,
            info.gradient_max_converged ? "✓" : "✗");
  log::info("  RMS step:        {:.5e} (threshold: {:.2e}) {}",
            info.rms_step, m_criteria.step_rms,
            info.step_rms_converged ? "✓" : "✗");
  log::info("  Max step:        {:.5e} (threshold: {:.2e}) {}",
            info.max_step, m_criteria.step_max,
            info.step_max_converged ? "✓" : "✗");
  if (m_criteria.use_energy_criterion && m_step > 0)
    log::info("  Energy change:   {:.5e} (threshold: {:.2e}) {}",
              info.energy_change, m_criteria.energy_change,
              info.energy_converged ? "✓" : "✗");
  if (info.converged && !rejected) {
    m_converged = true;
    log::info("Optimization converged!");
    return true;
  }

  m_predicted_change = gd * (scale - 0.5 * scale * scale);
  m_last_disp = max_disp;
  m_x_prev = m_x;
  m_g_prev = m_g;
  m_energy_prev = m_energy;
  m_x += d;
  m_step++;
  return false;
}

core::Molecule LbfgsOptimizer::get_next_geometry() const {
  Mat3N pos = m_x.reshaped(3, m_x.size() / 3) * occ::units::BOHR_TO_ANGSTROM;
  return core::Molecule(m_molecule.atomic_numbers(), pos);
}

} // namespace occ::opt
