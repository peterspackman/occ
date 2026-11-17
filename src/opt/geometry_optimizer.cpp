#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <occ/core/eigh.h>
#include <occ/core/log.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/opt/model_hessian.h>
#include <stdexcept>

namespace occ::opt {

namespace {

// Moore-Penrose inverse of B from the eigendecomposition of B^T B, keeping
// directions above 1e-8 of the largest, and an orthonormal basis of the
// range of B (the non-redundant, delocalised combinations of coordinates).
struct Inverse {
  Mat inverse; // 3N x m
  Mat range;   // m x rank
};

Inverse invert(const Mat &B) {
  auto [ev, V] = occ::core::eigh(B.transpose() * B);
  const double cutoff = 1e-8 * ev.maxCoeff();
  int rank = 0;
  for (int i = 0; i < ev.size(); i++)
    rank += ev(i) > cutoff;
  const Mat Vr = V.rightCols(rank);
  const Vec d = ev.tail(rank);
  return {Vr * d.cwiseInverse().asDiagonal() * (Vr.transpose() * B.transpose()),
          B * (Vr * d.cwiseSqrt().cwiseInverse().asDiagonal())};
}

// Geometry whose coordinates differ from those at x0 by dq: damped Newton
// iterations on q(x) = q(x0) + dq, keeping the best iterate. Empty if that
// misses dq by more than 0.1 (geomeTRIC's threshold).
std::optional<Mat3N> back_transform(const CoordinateSystem &coords,
                                    const Mat3N &x0, const Vec &dq) {
  const Vec q0 = coords.values(x0);
  Mat3N x = x0, best_x = x0;
  Vec residual = dq;
  double best = residual.norm(), previous = best, damping = 1.0;
  int failures = 0;
  for (int iteration = 0; iteration < 50 && failures < 5; iteration++) {
    const Mat3N dx = damping * (invert(coords.wilson_b(x)).inverse * residual)
                                   .reshaped(3, x.cols());
    x += dx;
    residual = dq - coords.difference(coords.values(x), q0);
    const double error = residual.norm();
    if (!std::isfinite(error))
      return std::nullopt;
    if (error < best) {
      best = error;
      best_x = x;
    }
    if (error > previous) {
      damping *= 0.5;
      failures++;
    } else {
      damping = std::min(1.2 * damping, 1.0);
    }
    previous = error;
    if (error < 1e-6 || std::sqrt(dx.squaredNorm() / x.cols()) < 1e-6)
      break;
  }
  if (best > 0.1)
    return std::nullopt;
  return best_x;
}

double rms(const Vec &v) {
  return v.size() ? std::sqrt(v.squaredNorm() / v.size()) : 0.0;
}

} // namespace

bool check_convergence(const Mat3N &gradient, const Vec &displacements,
                       double energy_change, bool check_energy,
                       const ConvergenceCriteria &criteria, int step) {
  const Vec g = gradient.colwise().norm().transpose();
  struct Criterion {
    const char *name;
    double value, threshold;
    bool active;
  };
  const Criterion checks[] = {
      {"RMS gradient", rms(g), criteria.gradient_rms, true},
      {"Max gradient", g.size() ? g.maxCoeff() : 0.0, criteria.gradient_max,
       true},
      {"RMS step", rms(displacements), criteria.step_rms, true},
      {"Max step", displacements.size() ? displacements.maxCoeff() : 0.0,
       criteria.step_max, true},
      {"Energy change", std::abs(energy_change), criteria.energy_change,
       check_energy}};
  bool converged = true;
  log::info("Step {:2d} convergence check (per atom):", step);
  for (const auto &c : checks) {
    if (!c.active)
      continue;
    const bool ok = c.value < c.threshold;
    converged = converged && ok;
    log::info("  {:<15s} {:.5e} (threshold: {:.2e}) {}", c.name, c.value,
              c.threshold, ok ? "✓" : "✗");
  }
  return converged;
}

Vec aligned_displacements(const Mat3N &x0, const Mat3N &x) {
  const Mat3N p0 = x0.colwise() - x0.rowwise().mean();
  const Mat3N p = x.colwise() - x.rowwise().mean();
  Eigen::JacobiSVD<Mat3> svd(p * p0.transpose(),
                             Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0
                ? -1.0
                : 1.0;
  const Mat3 R = svd.matrixU() * D * svd.matrixV().transpose();
  return (p - R * p0).colwise().norm().transpose();
}

Mat3N project_net_force_torque(const Mat3N &x, const Mat3N &gradient) {
  Mat3N g = gradient.colwise() - gradient.rowwise().mean();
  const Mat3N r = x.colwise() - x.rowwise().mean();
  Vec3 torque = Vec3::Zero();
  Mat3 inertia = Mat3::Zero();
  for (int i = 0; i < x.cols(); i++) {
    torque += r.col(i).cross(g.col(i));
    inertia += r.col(i).squaredNorm() * Mat3::Identity() -
               r.col(i) * r.col(i).transpose();
  }
  // least-squares rigid rotation in g; pseudo-inverse for linear systems
  const Vec3 omega = inertia.completeOrthogonalDecomposition().solve(torque);
  for (int i = 0; i < x.cols(); i++)
    g.col(i) -= omega.cross(r.col(i));
  return g;
}

GeometryOptimizer::GeometryOptimizer(const core::Molecule &molecule,
                                     const ConvergenceCriteria &criteria,
                                     const OptimizerSettings &settings)
    : m_atomic_numbers(molecule.atomic_numbers()), m_criteria(criteria),
      m_settings(settings), m_trust(settings.trust) {
  m_next = molecule.positions() * occ::units::ANGSTROM_TO_BOHR;
  build_coordinates(m_next, lindh_model_hessian(molecule.positions(),
                                                 m_atomic_numbers));
  log::info("Geometry optimizer: {} coordinates ({})",
            settings.coordinates == OptimizerSettings::Coordinates::TRIC
                ? "TRIC"
                : "Cartesian",
            m_coordinates->size());
}

void GeometryOptimizer::build_coordinates(const Mat3N &x,
                                          const Mat &cartesian_hessian) {
  if (m_settings.coordinates == OptimizerSettings::Coordinates::TRIC)
    m_coordinates = std::make_unique<TRICCoordinates>(m_atomic_numbers, x);
  else
    m_coordinates = std::make_unique<CartesianCoordinates>(x.cols());
  const Mat B_inv = invert(m_coordinates->wilson_b(x)).inverse;
  m_hessian = B_inv.transpose() * cartesian_hessian * B_inv;
  const Vec floor = m_coordinates->hessian_floor();
  for (int i = 0; i < floor.size(); i++)
    m_hessian(i, i) = std::max(m_hessian(i, i), floor(i));
}

void GeometryOptimizer::set_initial_hessian(const Mat &cartesian_hessian) {
  const Mat3N &x = m_accepted ? m_accepted->x : m_next;
  if (cartesian_hessian.rows() != x.size() ||
      cartesian_hessian.cols() != x.size())
    throw std::invalid_argument(
        "GeometryOptimizer: the Hessian must be 3N x 3N");
  build_coordinates(x, cartesian_hessian);
}

void GeometryOptimizer::update(double energy, const Mat3N &gradient) {
  if (gradient.rows() != 3 || gradient.cols() != m_next.cols())
    throw std::invalid_argument(
        "GeometryOptimizer::update(): gradient must be 3 x N atoms");
  m_latest = Point{m_next, energy,
                   m_settings.project_net_force_torque
                       ? project_net_force_torque(m_next, gradient)
                       : gradient};
}

void GeometryOptimizer::shift_energy_reference(double shift) {
  if (m_accepted)
    m_accepted->energy += shift;
  m_previous_accepted_energy += shift;
}

// BFGS between two points, both in the current coordinates; skipped when it
// would lose positive definiteness.
void GeometryOptimizer::update_hessian(const Point &from, const Point &to) {
  const auto &c = *m_coordinates;
  const Vec s = c.difference(c.values(to.x), c.values(from.x));
  const Vec y =
      invert(c.wilson_b(to.x)).inverse.transpose() * to.gradient.reshaped() -
      invert(c.wilson_b(from.x)).inverse.transpose() *
          from.gradient.reshaped();
  const Vec Hs = m_hessian * s;
  const double sy = s.dot(y), sHs = s.dot(Hs);
  if (sy <= 1e-12 * s.norm() * y.norm() || sHs <= 0.0) {
    log::debug("BFGS update skipped: s.y = {:.3e}", sy);
    return;
  }
  m_hessian += y * y.transpose() / sy - Hs * Hs.transpose() / sHs;
}

// The step from p: the Newton step -(H + v)^-1 g in the delocalised
// coordinates, with v >= 0 making H + v positive definite, taken whole if it
// moves the atoms (RMS, aligned) no further than the trust radius, otherwise
// shortened by a larger v until it does. If no step can be back-transformed
// the coordinates are rebuilt; as a last resort, a steepest descent step in
// Cartesians.
GeometryOptimizer::Step GeometryOptimizer::next_step(const Point &p) {
  const Vec g = p.gradient.reshaped();
  for (int attempt = 0; attempt < 2; attempt++) {
    if (attempt == 1) {
      log::info("Rebuilding coordinates: no step could be back-transformed");
      const Mat B = m_coordinates->wilson_b(p.x);
      build_coordinates(p.x, B.transpose() * m_hessian * B);
    }
    const auto [B_inv, U] = invert(m_coordinates->wilson_b(p.x));
    const Vec g_q = B_inv.transpose() * g;
    auto [ev, V] = occ::core::eigh(U.transpose() * m_hessian * U);
    const Vec gt = V.transpose() * (U.transpose() * g_q);
    const double v0 = std::max(0.0, 1e-5 - ev(0));
    auto newton = [&](double v) -> Vec {
      return -(gt.array() / (ev.array() + v)).matrix();
    };
    // the shift giving a step of length r in the delocalised coordinates
    auto shift_for = [&](double r) {
      double lo = v0, hi = v0 + 1.0;
      while (newton(hi).norm() > r)
        hi = v0 + 2.0 * (hi - v0);
      for (int i = 0; i < 100 && hi - lo > 1e-12 * (1.0 + hi); i++) {
        const double mid = 0.5 * (lo + hi);
        (newton(mid).norm() > r ? lo : hi) = mid;
      }
      return hi;
    };
    auto trial = [&](double v) -> std::optional<Step> {
      const Vec dq = U * (V * newton(v));
      auto x = back_transform(*m_coordinates, p.x, dq);
      if (!x)
        return std::nullopt;
      const auto &c = *m_coordinates;
      const Vec dy = c.difference(c.values(*x), c.values(p.x));
      return Step{*x, g_q.dot(dy) + 0.5 * dy.dot(m_hessian * dy),
                  rms(aligned_displacements(p.x, *x))};
    };

    auto full = trial(v0);
    if (full && full->rmsd <= 1.1 * m_trust)
      return *full;
    // bisect on the length in the coordinates for an RMS displacement within
    // 10% of the trust radius, keeping the longest step inside it
    std::optional<Step> best;
    double lo = 0.0, hi = newton(v0).norm();
    for (int i = 0; i < 30; i++) {
      const double r = 0.5 * (lo + hi);
      auto s = trial(shift_for(r));
      if (!s || s->rmsd > 1.1 * m_trust) {
        hi = r;
        continue;
      }
      lo = r;
      best = s;
      if (s->rmsd > 0.9 * m_trust)
        break;
    }
    if (best)
      return *best;
  }
  log::warn("No internal coordinate step could be taken; "
            "taking a steepest descent step in Cartesians");
  const double scale = m_trust * std::sqrt(double(p.x.cols())) /
                       std::max(g.norm(), 1e-300);
  const Mat3N x = p.x - scale * p.gradient;
  return {x, -scale * g.squaredNorm(), m_trust};
}

bool GeometryOptimizer::step() {
  if (!m_latest)
    throw std::runtime_error("GeometryOptimizer::step(): call update() first");
  const Point latest = *m_latest;
  m_latest.reset();

  bool rejected = false;
  if (!m_accepted) {
    m_accepted = latest;
    m_previous_accepted_energy = latest.energy;
  } else {
    const double actual = latest.energy - m_accepted->energy;
    const double predicted = m_last_step->predicted;
    const double quality = predicted < 0.0 ? actual / predicted : 1.0;
    const double reduced = std::max(
        m_settings.trust_min, 0.5 * std::min(m_trust, m_last_step->rmsd));
    // geomeTRIC's exemptions keep noisy energies from rejecting forever
    const bool repeated =
        m_last_rejected &&
        (latest.x - *m_last_rejected).cwiseAbs().maxCoeff() < 1e-6;
    if (quality < m_settings.reject_quality && actual > 0.0 &&
        m_trust > m_settings.trust_min &&
        m_last_step->rmsd > 1.2 * m_settings.trust_min &&
        actual > m_criteria.energy_change && !repeated) {
      m_trust = reduced;
      m_last_rejected = latest.x;
      rejected = true;
      log::info("Step rejected: energy rose by {:.3e} Ha (quality {:.2f}); "
                "trust radius now {:.3e} Bohr",
                actual, quality, m_trust);
    } else {
      update_hessian(*m_accepted, latest);
      if (quality > 0.75)
        m_trust = std::min(m_settings.trust_max, std::sqrt(2.0) * m_trust);
      else if (quality < 0.25)
        m_trust = reduced;
      log::debug("Step quality {:.3f}; trust radius {:.3e} Bohr", quality,
                 m_trust);
      m_previous_accepted_energy = m_accepted->energy;
      m_accepted = latest;
    }
  }

  const Point &p = *m_accepted;
  if (m_coordinates->degenerate(p.x)) {
    log::info("Rebuilding coordinates at the current geometry");
    const Mat B = m_coordinates->wilson_b(p.x);
    build_coordinates(p.x, B.transpose() * m_hessian * B);
  }

  const Step next = next_step(p);
  const bool converged = check_convergence(
      p.gradient, aligned_displacements(p.x, next.x),
      p.energy - m_previous_accepted_energy,
      m_criteria.use_energy_criterion && m_step > 0, m_criteria, m_step);

  // Not right after a rejection: the smaller trust radius alone could pass
  // the step criteria, and the point last evaluated is not this one.
  if (converged && !rejected) {
    m_converged = true;
    m_next = p.x;
    log::info("Optimization converged!");
    return true;
  }
  m_next = next.x;
  m_last_step = next;
  m_step++;
  return false;
}

core::Molecule GeometryOptimizer::get_next_geometry() const {
  return core::Molecule(m_atomic_numbers,
                        m_next * occ::units::BOHR_TO_ANGSTROM);
}

} // namespace occ::opt
