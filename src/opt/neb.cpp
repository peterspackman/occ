#include <Eigen/SVD>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/opt/neb.h>
#include <stdexcept>
#include <tuple>

namespace occ::opt {

namespace {

// x moved onto x0 by the translation and proper rotation that superpose
// them best (Kabsch).
Mat3N superpose(const Mat3N &x0, const Mat3N &x) {
  const Vec3 c0 = x0.rowwise().mean(), c = x.rowwise().mean();
  const Mat3N p0 = x0.colwise() - c0, p = x.colwise() - c;
  Eigen::JacobiSVD<Mat3> svd(p0 * p.transpose(),
                             Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0
                ? -1.0
                : 1.0;
  const Mat3 R = svd.matrixU() * D * svd.matrixV().transpose();
  return (R * p).colwise() + c0;
}

// The image-dependent pair potential: sum over pairs of
// (d - D)^2 / d^4, D the target distance.
std::pair<double, Mat3N> idpp(const Mat3N &x, const Mat &target) {
  const int n = x.cols();
  double e = 0.0;
  Mat3N g = Mat3N::Zero(3, n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      const Vec3 r = x.col(i) - x.col(j);
      const double d = r.norm(), u = d - target(i, j), w = 1.0 / std::pow(d, 4);
      e += w * u * u;
      const double de = 2.0 * w * u - 4.0 * w * u * u / d;
      g.col(i) += de * r / d;
      g.col(j) -= de * r / d;
    }
  }
  return {e, g};
}

Mat distances(const Mat3N &x) {
  const int n = x.cols();
  Mat d(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      d(i, j) = (x.col(i) - x.col(j)).norm();
  return d;
}

// largest per-atom norm of a stacked 3N x images force
double largest_atom_force(const Vec &f) {
  return f.size() ? f.reshaped(3, f.size() / 3).colwise().norm().maxCoeff()
                  : 0.0;
}

} // namespace

NudgedElasticBand::NudgedElasticBand(const core::Molecule &reactant,
                                     const core::Molecule &product,
                                     const NebSettings &settings)
    : m_atomic_numbers(reactant.atomic_numbers()), m_settings(settings) {
  if (product.atomic_numbers() != reactant.atomic_numbers())
    throw std::invalid_argument("NudgedElasticBand: the reactant and product "
                                "must have the same atoms in the same order");
  if (settings.images < 1)
    throw std::invalid_argument("NudgedElasticBand: at least one image");
  const Mat3N x0 = reactant.positions() * occ::units::ANGSTROM_TO_BOHR;
  const Mat3N x1 =
      superpose(x0, product.positions() * occ::units::ANGSTROM_TO_BOHR);
  const int m = settings.images + 1;
  for (int k = 0; k <= m; k++) {
    const double t = double(k) / m;
    m_x.push_back((1.0 - t) * x0 + t * x1);
  }
  m_energy = Vec::Zero(m + 1);
  if (settings.idpp)
    interpolate_idpp();
}

// Relax the straight line on the pair potential whose target distances
// interpolate those of the endpoints, as a band of its own. FIRE (Bitzek et
// al., Phys. Rev. Lett. 97, 170201 (2006)) moves it: the potential is stiff
// (1/d^4 where the line takes atoms through each other) and evaluations
// cost nothing, where L-BFGS on the band's non-conservative force oscillated.
void NudgedElasticBand::interpolate_idpp() {
  const NebSettings saved = m_settings;
  m_settings.climbing_image = false;
  m_settings.project_net_force_torque = false;
  m_settings.spring_max = m_settings.spring;
  const Mat d0 = distances(m_x.front()), d1 = distances(m_x.back());
  const int m = m_x.size() - 1, n = m - 1, size = m_x[0].size();
  set_endpoint_energies(0.0, 0.0);

  constexpr double tolerance = 1e-4, dt_max = 1.0, alpha0 = 0.1;
  double dt = 0.1, alpha = alpha0, force_max = 0.0;
  int downhill = 0, iteration = 0;
  Vec v = Vec::Zero(n * size);
  for (; iteration < 10000; iteration++) {
    std::vector<double> e;
    std::vector<Mat3N> g;
    for (int k = 1; k < m; k++) {
      const double t = double(k) / m;
      auto [ek, gk] = idpp(m_x[k], (1.0 - t) * d0 + t * d1);
      e.push_back(ek);
      g.push_back(gk);
    }
    update(e, g);
    const Vec f = neb_force();
    force_max = largest_atom_force(f);
    if (force_max < tolerance)
      break;
    if (f.dot(v) > 0.0) {
      v = (1.0 - alpha) * v + alpha * v.norm() * f.normalized();
      if (++downhill > 5) {
        dt = std::min(1.1 * dt, dt_max);
        alpha *= 0.99;
      }
    } else {
      v.setZero();
      dt *= 0.5;
      alpha = alpha0;
      downhill = 0;
    }
    v += dt * f;
    Vec dx = dt * v;
    const double largest = largest_atom_force(dx);
    if (largest > m_settings.max_step)
      dx *= m_settings.max_step / largest;
    for (int k = 1; k <= n; k++)
      m_x[k] += dx.segment(size * (k - 1), size).reshaped(3, m_x[0].cols());
  }
  if (force_max < tolerance)
    log::debug("IDPP interpolation: {} iterations", iteration + 1);
  else
    log::warn("IDPP interpolation did not settle (largest force {:.2e}); "
              "starting from it anyway",
              force_max);
  m_settings = saved;
  m_have_endpoints = m_have_update = false;
}

std::vector<core::Molecule> NudgedElasticBand::images() const {
  std::vector<core::Molecule> result;
  for (size_t k = 1; k + 1 < m_x.size(); k++)
    result.emplace_back(m_atomic_numbers,
                        m_x[k] * occ::units::BOHR_TO_ANGSTROM);
  return result;
}

std::vector<core::Molecule> NudgedElasticBand::path() const {
  std::vector<core::Molecule> result;
  for (const auto &x : m_x)
    result.emplace_back(m_atomic_numbers, x * occ::units::BOHR_TO_ANGSTROM);
  return result;
}

void NudgedElasticBand::set_endpoint_energies(double reactant, double product) {
  m_energy(0) = reactant;
  m_energy(m_energy.size() - 1) = product;
  m_have_endpoints = true;
}

void NudgedElasticBand::update(const std::vector<double> &energies,
                               const std::vector<Mat3N> &gradients) {
  const size_t n = m_x.size() - 2;
  if (energies.size() != n || gradients.size() != n)
    throw std::invalid_argument(
        "NudgedElasticBand::update(): one energy and gradient per image");
  m_gradient.clear();
  for (size_t k = 0; k < n; k++) {
    if (gradients[k].rows() != 3 || gradients[k].cols() != m_x[0].cols())
      throw std::invalid_argument(
          "NudgedElasticBand::update(): gradients must be 3 x N atoms");
    m_energy(k + 1) = energies[k];
    m_gradient.push_back(m_settings.project_net_force_torque
                             ? project_net_force_torque(m_x[k + 1], gradients[k])
                             : gradients[k]);
  }
  m_have_update = true;
}

int NudgedElasticBand::highest_image() const {
  int best = 1;
  for (int k = 2; k + 1 < m_energy.size(); k++)
    if (m_energy(k) > m_energy(best))
      best = k;
  return best;
}

// Differences to the neighbours of image i, each superposed onto image i
// first, so that a neighbour's orientation adds no rigid rotation to the
// tangent or the spring lengths.
std::pair<Vec, Vec> NudgedElasticBand::neighbour_differences(int i) const {
  return {(superpose(m_x[i], m_x[i + 1]) - m_x[i]).reshaped(),
          (m_x[i] - superpose(m_x[i], m_x[i - 1])).reshaped()};
}

// The improved tangent: towards the higher neighbour, or at an extremum the
// energy-weighted mix of both directions.
Vec NudgedElasticBand::tangent(int i) const {
  const auto [plus, minus] = neighbour_differences(i);
  const double ep = m_energy(i + 1), e = m_energy(i), em = m_energy(i - 1);
  Vec t;
  if (ep > e && e > em) {
    t = plus;
  } else if (ep < e && e < em) {
    t = minus;
  } else {
    const double big = std::max(std::abs(ep - e), std::abs(em - e));
    const double small = std::min(std::abs(ep - e), std::abs(em - e));
    t = ep > em ? (big * plus + small * minus).eval()
                : (small * plus + big * minus).eval();
  }
  const double norm = t.norm();
  return norm > 0.0 ? (t / norm).eval() : plus.normalized();
}

// Stacked NEB forces on the interior images: the potential force
// perpendicular to the tangent and the spring force along it, or for the
// climbing image the potential force with its component along the tangent
// reversed.
Vec NudgedElasticBand::neb_force() const {
  const int n = m_x.size() - 2, size = m_x[0].size();
  const int climber = m_climbing ? highest_image() : -1;
  // the spring between images k and k+1, by the higher of their energies
  const double e_ref = std::max(m_energy(0), m_energy(n + 1));
  const double e_max = m_energy.maxCoeff();
  auto spring = [&](int k) {
    const double e = std::max(m_energy(k), m_energy(k + 1));
    if (e <= e_ref || e_max <= e_ref)
      return m_settings.spring;
    return m_settings.spring_max - (m_settings.spring_max - m_settings.spring) *
                                       (e_max - e) / (e_max - e_ref);
  };
  Vec f(n * size);
  for (int k = 1; k <= n; k++) {
    const Vec t = tangent(k);
    const Vec g = m_gradient[k - 1].reshaped();
    Vec fk;
    if (k == climber) {
      fk = -g + 2.0 * g.dot(t) * t;
    } else {
      const auto [plus, minus] = neighbour_differences(k);
      const double stretch =
          spring(k) * plus.norm() - spring(k - 1) * minus.norm();
      fk = -(g - g.dot(t) * t) + stretch * t;
    }
    f.segment(size * (k - 1), size) = fk;
  }
  return f;
}

// Clear the L-BFGS history.
void NudgedElasticBand::reset_history() {
  m_s.clear();
  m_y.clear();
}

// L-BFGS on the band: the force is the negative of a gradient (though of no
// energy). A trust radius on the largest atomic displacement moves on the
// force alone: a step that more than doubles the largest force is undone
// (the band goes back to the last accepted one, whose energies and gradients
// are kept) and the radius halves; steps that do not grow the force let it
// grow back. History is cleared on a rejection, when a step is capped (as
// ORCA does), and when the climbing image changes.
bool NudgedElasticBand::step() {
  if (!m_have_endpoints || !m_have_update)
    throw std::runtime_error("NudgedElasticBand::step(): set the endpoint "
                             "energies and call update() first");
  m_have_update = false;
  const int n = m_x.size() - 2, size = m_x[0].size();
  auto stacked = [&]() {
    Vec x(n * size);
    for (int k = 1; k <= n; k++)
      x.segment(size * (k - 1), size) = m_x[k].reshaped();
    return x;
  };
  // the largest force on any atom of the climbing image and of the others
  auto largest_forces = [&](const Vec &f) {
    const int climber = m_climbing ? highest_image() : -1;
    double image = 0.0, climbing = 0.0;
    for (int k = 1; k <= n; k++) {
      const double fk = largest_atom_force(f.segment(size * (k - 1), size));
      (k == climber ? climbing : image) =
          std::max(k == climber ? climbing : image, fk);
    }
    return std::pair{image, climbing};
  };
  if (m_trust <= 0.0)
    m_trust = m_settings.max_step;

  Vec x = stacked();
  Vec force = neb_force();
  auto [image_force, climbing_force] = largest_forces(force);
  m_max_force = std::max(image_force, climbing_force);
  log::debug("NEB step {}: largest force {:.3e} Ha/Bohr, trust {:.3e} Bohr{}",
             m_step, m_max_force, m_trust, m_climbing ? " (climbing)" : "");

  const bool have_accepted = m_accepted_x.size() == x.size();
  if (have_accepted && m_max_force > 2.0 * m_accepted_max_force &&
      m_trust > 1e-3) {
    log::debug("NEB: step rejected (largest force {:.3e} from {:.3e})",
               m_max_force, m_accepted_max_force);
    for (int k = 1; k <= n; k++)
      m_x[k] = m_accepted_x.segment(size * (k - 1), size).reshaped(3, m_x[0].cols());
    m_energy = m_accepted_energy;
    m_gradient = m_accepted_gradient;
    x = m_accepted_x;
    force = neb_force();
    std::tie(image_force, climbing_force) = largest_forces(force);
    m_max_force = std::max(image_force, climbing_force);
    m_trust *= 0.5;
    reset_history();
  } else {
    if (have_accepted) {
      const Vec s = x - m_accepted_x, y = m_accepted_force - force;
      if (s.dot(y) > 1e-12 * s.norm() * y.norm()) {
        m_s.push_back(s);
        m_y.push_back(y);
        if (int(m_s.size()) > m_settings.memory) {
          m_s.pop_front();
          m_y.pop_front();
        }
      }
      if (m_max_force <= m_accepted_max_force)
        m_trust = std::min(m_settings.max_step, 1.2 * m_trust);
    }
    // another image has become the highest: its force changes definition,
    // so the history no longer describes one surface
    if (m_climbing && highest_image() != m_climber) {
      log::debug("NEB: image {} climbs in place of {}", highest_image(),
                 m_climber);
      m_climber = highest_image();
      reset_history();
    }
    if (m_settings.climbing_image && !m_climbing &&
        m_max_force < m_settings.climb_threshold) {
      m_climbing = true;
      m_climber = highest_image();
      log::info("NEB: image {} climbs", m_climber);
      reset_history();
      force = neb_force();
      std::tie(image_force, climbing_force) = largest_forces(force);
      m_max_force = std::max(image_force, climbing_force);
    }
    if (image_force < m_settings.force_max &&
        climbing_force < m_settings.climbing_force_max &&
        (m_climbing || !m_settings.climbing_image)) {
      m_converged = true;
      return true;
    }
    m_accepted_x = x;
    m_accepted_force = force;
    m_accepted_max_force = m_max_force;
    m_accepted_energy = m_energy;
    m_accepted_gradient = m_gradient;
  }

  // the L-BFGS direction from the accepted band
  Vec d = force;
  const int m = m_s.size();
  std::vector<double> alpha(m);
  for (int i = m - 1; i >= 0; i--) {
    alpha[i] = m_s[i].dot(d) / m_y[i].dot(m_s[i]);
    d -= alpha[i] * m_y[i];
  }
  if (m > 0)
    d *= m_s.back().dot(m_y.back()) / m_y.back().squaredNorm();
  for (int i = 0; i < m; i++) {
    const double beta = m_y[i].dot(d) / m_y[i].dot(m_s[i]);
    d += (alpha[i] - beta) * m_s[i];
  }
  if (d.dot(force) <= 0.0) {
    log::debug("NEB: L-BFGS direction against the force; history cleared");
    reset_history();
    d = force;
  }
  const double largest = largest_atom_force(d);
  if (largest > m_trust) {
    d *= m_trust / largest;
    reset_history();
  }
  x += d;
  for (int k = 1; k <= n; k++)
    m_x[k] = x.segment(size * (k - 1), size).reshaped(3, m_x[0].cols());
  m_step++;
  return false;
}

} // namespace occ::opt
