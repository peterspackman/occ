#include <Eigen/SVD>
#include <cmath>
#include <functional>
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
  build_coordinates(m_next,
                    lindh_model_hessian(molecule.positions(), m_atomic_numbers),
                    true);
  log::info("Geometry optimizer: {} coordinates ({})",
            settings.coordinates == OptimizerSettings::Coordinates::TRIC
                ? "TRIC"
                : "Cartesian",
            m_coordinates->size());
  if (settings.transition_state)
    log::info("Searching for a first-order saddle point, starting along "
              "internal mode {} from the lowest",
              settings.ts_mode);
}

// The Hessian in new coordinates at x from a Cartesian one. The model's gets
// least stiffnesses (Hessian floors); a computed or carried one is kept as it
// is, negative curvature and all.
void GeometryOptimizer::build_coordinates(const Mat3N &x,
                                          const Mat &cartesian_hessian,
                                          bool model) {
  if (m_settings.coordinates == OptimizerSettings::Coordinates::TRIC)
    m_coordinates = std::make_unique<TRICCoordinates>(m_atomic_numbers, x);
  else
    m_coordinates = std::make_unique<CartesianCoordinates>(x.cols());
  const Mat B_inv = invert(m_coordinates->wilson_b(x)).inverse;
  m_hessian = B_inv.transpose() * cartesian_hessian * B_inv;
  if (!model)
    return;
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
  Mat H = 0.5 * (cartesian_hessian + cartesian_hessian.transpose());
  // Away from a stationary point the Hessian has curvature, possibly
  // negative, along rigid rotations, which leave an isolated system's energy
  // unchanged. It is replaced by a small stiffness (the model's least for
  // rigid-body coordinates): a saddle-point step would chase negative
  // curvature there, and with none at all the residual gradient along it
  // (numerical noise) would give a step without bound.
  if (m_settings.project_net_force_torque) {
    constexpr double rigid_stiffness = 0.01; // Hartree/Bohr^2
    const Mat Q = rigid_body_basis(x);
    const Mat HQ = H * Q;
    H -= HQ * Q.transpose() + Q * HQ.transpose() -
         Q * (Q.transpose() * HQ) * Q.transpose();
    H += rigid_stiffness * Q * Q.transpose();
  }
  build_coordinates(x, H, false);
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

// BFGS between two points, both in the current coordinates, skipped when it
// would lose positive definiteness; for a saddle point Bofill's update, the
// SR1 (Murtagh-Sargent) and Powell-symmetric-Broyden updates mixed by the
// squared cosine of the angle between the step and the secant error.
void GeometryOptimizer::update_hessian(const Point &from, const Point &to) {
  const auto &c = *m_coordinates;
  const Vec s = c.difference(c.values(to.x), c.values(from.x));
  const Vec y =
      invert(c.wilson_b(to.x)).inverse.transpose() * to.gradient.reshaped() -
      invert(c.wilson_b(from.x)).inverse.transpose() *
          from.gradient.reshaped();
  if (m_settings.transition_state) {
    const Vec xi = y - m_hessian * s;
    const double ss = s.squaredNorm(), xx = xi.squaredNorm(), sx = s.dot(xi);
    if (ss < 1e-20 || xx < 1e-24)
      return;
    const double phi = 1.0 - sx * sx / (ss * xx);
    const Mat psb = (xi * s.transpose() + s * xi.transpose()) / ss -
                    sx * s * s.transpose() / (ss * ss);
    m_hessian += phi * psb;
    if (phi < 1.0)
      m_hessian += (1.0 - phi) * xi * xi.transpose() / sx;
    return;
  }
  const Vec Hs = m_hessian * s;
  const double sy = s.dot(y), sHs = s.dot(Hs);
  if (sy <= 1e-12 * s.norm() * y.norm() || sHs <= 0.0) {
    log::debug("BFGS update skipped: s.y = {:.3e}", sy);
    return;
  }
  m_hessian += y * y.transpose() / sy - Hs * Hs.transpose() / sHs;
}

// The mode to maximise along, among the eigenvectors (columns of `modes`, as
// Cartesian displacements at x, by ascending curvature) that are mostly
// internal motion: at the first step the requested one counting from the
// lowest, later the one overlapping most with the mode followed so far.
int GeometryOptimizer::follow_mode(const Mat3N &x, const Mat &modes,
                                   const Vec &curvature) {
  const Mat Q = rigid_body_basis(x);
  std::vector<int> internal;
  for (int i = 0; i < modes.cols(); i++) {
    const double norm = modes.col(i).norm();
    if (norm > 0.0 &&
        (modes.col(i) - Q * (Q.transpose() * modes.col(i))).norm() >
            0.5 * norm)
      internal.push_back(i);
  }
  if (internal.empty())
    throw std::runtime_error("GeometryOptimizer: no internal mode to follow");
  int t = internal.front();
  if (m_mode.size() == 0) {
    t = internal[std::min<int>(m_settings.ts_mode, internal.size() - 1)];
  } else {
    // the most overlapping mode, among those with negative curvature if the
    // most overlapping one overall has none: maximising along a stiff mode
    // climbs it without end
    auto most_overlapping = [&](bool negative_only) {
      int best_i = -1;
      double best = -1.0;
      for (int i : internal) {
        if (negative_only && curvature(i) >= 0.0)
          continue;
        const double overlap =
            std::abs(modes.col(i).dot(m_mode)) / modes.col(i).norm();
        if (overlap > best) {
          best = overlap;
          best_i = i;
        }
      }
      return std::pair{best_i, best};
    };
    auto [i, overlap] = most_overlapping(false);
    if (curvature(i) >= 0.0) {
      const auto [j, overlap_j] = most_overlapping(true);
      if (j >= 0) {
        log::info("The followed mode has lost its negative curvature; "
                  "following the negative mode overlapping it most ({:.2f})",
                  overlap_j);
        i = j;
        overlap = overlap_j;
      }
    }
    t = i;
    log::debug("Followed mode: curvature {:.4e}, overlap {:.3f}",
               curvature(t), overlap);
  }
  m_negative_modes = 0;
  for (int i : internal)
    m_negative_modes += curvature(i) < 0.0;
  Vec mode = modes.col(t).normalized();
  if (m_mode.size() == mode.size() && mode.dot(m_mode) < 0.0)
    mode = -mode;
  m_mode = mode;
  return t;
}

// The step from p in the delocalised coordinates (the range of B), within a
// length r there. For a minimum, the Newton step -(H + v)^-1 g with v >= 0
// making H + v positive definite, v raised until the step is short enough.
// For a saddle point, the restricted-step partitioned RFO step (Bofill,
// Int. J. Quantum Chem. 94, 324 (2003)): along the followed mode t the
// energy is maximised, -g_t / (h_t - mu_p), along the others minimised,
// -g_i / (h_i - mu_n), with mu_p and mu_n the largest and least eigenvalues
// of the two augmented Hessians in the metric diag(alpha, ..., 1), alpha >= 1
// raised until the step is short enough.
//
// The step taken is the whole one if it moves the atoms (RMS, aligned) no
// further than the trust radius, otherwise the one whose length in the
// coordinates gives a displacement within 10% of it. If no step can be
// back-transformed the coordinates are rebuilt; as a last resort, a steepest
// descent step in Cartesians.
GeometryOptimizer::Step GeometryOptimizer::next_step(const Point &p) {
  const Vec g = p.gradient.reshaped();
  const bool saddle = m_settings.transition_state;
  for (int attempt = 0; attempt < 2; attempt++) {
    if (attempt == 1) {
      log::info("Rebuilding coordinates: no step could be back-transformed");
      const Mat B = m_coordinates->wilson_b(p.x);
      build_coordinates(p.x, B.transpose() * m_hessian * B, !saddle);
    }
    const auto [B_inv, U] = invert(m_coordinates->wilson_b(p.x));
    const Vec g_q = B_inv.transpose() * g;
    auto [ev, V] = occ::core::eigh(U.transpose() * m_hessian * U);
    const Vec gt = V.transpose() * (U.transpose() * g_q);
    const int n = ev.size();

    // the step, as a function of the parameter that shortens it
    std::function<Vec(double)> step_for;
    double parameter0 = 0.0;
    if (!saddle) {
      parameter0 = std::max(0.0, 1e-5 - ev(0));
      step_for = [&](double v) -> Vec {
        return -(gt.array() / (ev.array() + v)).matrix();
      };
    } else {
      const int t = follow_mode(p.x, B_inv * (U * V), ev);
      double h_min = std::numeric_limits<double>::infinity();
      for (int i = 0; i < n; i++)
        if (i != t)
          h_min = std::min(h_min, ev(i));
      const double upper = std::min(0.0, h_min);
      parameter0 = 1.0;
      step_for = [&, t, upper](double alpha) -> Vec {
        Vec s = Vec::Zero(n);
        const double mu_p =
            0.5 * (ev(t) + std::sqrt(ev(t) * ev(t) + 4.0 * alpha * gt(t) * gt(t)));
        if (std::abs(mu_p - ev(t)) > 1e-14)
          s(t) = gt(t) / (mu_p - ev(t));
        // mu_n / alpha = sum_i g_i^2 / (mu_n - h_i), the root below upper
        auto f = [&](double mu) {
          double r = mu / alpha;
          for (int i = 0; i < n; i++)
            if (i != t)
              r -= gt(i) * gt(i) / (mu - ev(i));
          return r;
        };
        double hi = upper, lo = upper - 1.0;
        while (f(lo) > 0.0)
          lo = upper - 2.0 * (upper - lo);
        for (int i = 0; i < 200 && hi - lo > 1e-15 * (1.0 + std::abs(lo)); i++) {
          const double mid = 0.5 * (lo + hi);
          (f(mid) > 0.0 ? hi : lo) = mid;
        }
        const double mu_n = 0.5 * (lo + hi);
        for (int i = 0; i < n; i++)
          if (i != t && std::abs(ev(i) - mu_n) > 1e-14)
            s(i) = -gt(i) / (ev(i) - mu_n);
        return s;
      };
    }
    // the step at most r long in the coordinates; longer parameters shorten
    // it (doubling, then bisection)
    auto step_within = [&](double r) -> Vec {
      const Vec full = step_for(parameter0);
      if (full.norm() <= r)
        return full;
      auto length = [&](double d) { return step_for(parameter0 + d).norm(); };
      double lo = 0.0, hi = 1.0;
      while (length(hi) > r && hi < 1e30)
        hi *= 2.0;
      for (int i = 0; i < 100 && hi - lo > 1e-12 * (1.0 + hi); i++) {
        const double mid = 0.5 * (lo + hi);
        (length(mid) > r ? lo : hi) = mid;
      }
      return step_for(parameter0 + hi);
    };
    auto trial = [&](double r) -> std::optional<Step> {
      const Vec dq = U * (V * step_within(r));
      auto x = back_transform(*m_coordinates, p.x, dq);
      if (!x)
        return std::nullopt;
      const auto &c = *m_coordinates;
      const Vec dy = c.difference(c.values(*x), c.values(p.x));
      return Step{*x, g_q.dot(dy) + 0.5 * dy.dot(m_hessian * dy),
                  rms(aligned_displacements(p.x, *x))};
    };

    const double full_length = step_for(parameter0).norm();
    auto full = trial(full_length);
    if (full && full->rmsd <= 1.1 * m_trust)
      return *full;
    // bisect on the length in the coordinates for an RMS displacement within
    // 10% of the trust radius, keeping the longest step inside it
    std::optional<Step> best;
    double lo = 0.0, hi = full_length;
    for (int i = 0; i < 30; i++) {
      const double r = 0.5 * (lo + hi);
      auto s = trial(r);
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
  if (saddle)
    throw std::runtime_error("GeometryOptimizer: no saddle-point step could "
                             "be taken in internal coordinates");
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
    const bool saddle = m_settings.transition_state;
    double quality = std::abs(predicted) > 1e-14 ? actual / predicted : 1.0;
    // near a saddle a model that underestimates the change is as wrong as
    // one that overestimates it
    if (saddle && quality > 1.0)
      quality = 2.0 - quality;
    const double reduced = std::max(
        m_settings.trust_min, 0.5 * std::min(m_trust, m_last_step->rmsd));
    // geomeTRIC's exemptions keep noisy energies from rejecting forever
    const bool repeated =
        m_last_rejected &&
        (latest.x - *m_last_rejected).cwiseAbs().maxCoeff() < 1e-6;
    const bool bad = saddle ? quality <= 0.0
                            : quality < m_settings.reject_quality && actual > 0.0;
    if (bad && std::abs(actual) > m_criteria.energy_change &&
        m_trust > m_settings.trust_min &&
        m_last_step->rmsd > 1.2 * m_settings.trust_min && !repeated) {
      m_trust = reduced;
      m_last_rejected = latest.x;
      rejected = true;
      log::info("Step rejected: energy changed by {:.3e} Ha (quality {:.2f}); "
                "trust radius now {:.3e} Bohr",
                actual, quality, m_trust);
    } else {
      update_hessian(*m_accepted, latest);
      if (quality > 0.75)
        m_trust = std::min(m_settings.trust_max, std::sqrt(2.0) * m_trust);
      else if (quality < (saddle ? 0.5 : 0.25))
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
    build_coordinates(p.x, B.transpose() * m_hessian * B,
                      !m_settings.transition_state);
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
    if (m_settings.transition_state && m_negative_modes != 1)
      log::warn("The model Hessian has {} negative internal modes at the "
                "converged point; check the frequencies",
                m_negative_modes);
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
