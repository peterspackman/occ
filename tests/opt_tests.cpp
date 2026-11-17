#include <Eigen/Eigenvalues>
#include <functional>
#include <catch2/catch_all.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/opt/lbfgs_optimizer.h>
#include <occ/opt/model_hessian.h>
#include <occ/opt/neb.h>

using namespace occ::opt;

// A model surface for optimizer tests: harmonic springs between atom pairs,
// energy in Hartree and gradient in Hartree/Bohr like the QM drivers.
namespace {
struct Spring {
  int i, j;
  double k;  // Hartree/Bohr^2
  double r0; // Bohr
};

std::pair<double, occ::Mat3N> spring_energy(const occ::core::Molecule &m,
                                            const std::vector<Spring> &springs) {
  const occ::Mat3N x = m.positions() * occ::units::ANGSTROM_TO_BOHR;
  occ::Mat3N g = occ::Mat3N::Zero(3, x.cols());
  double e = 0.0;
  for (const auto &s : springs) {
    const occ::Vec3 d = x.col(s.i) - x.col(s.j);
    const double r = d.norm();
    e += 0.5 * s.k * (r - s.r0) * (r - s.r0);
    const occ::Vec3 f = s.k * (r - s.r0) * d / r;
    g.col(s.i) += f;
    g.col(s.j) -= f;
  }
  return {e, g};
}

// The optimizer loop of the geometry optimization driver on a surface
// (energy, gradient) = surface(molecule); returns the number of gradient
// evaluations.
template <typename Opt, typename Surface>
int optimize(Opt &opt, const Surface &surface, int max_steps = 300) {
  for (int n = 1; n <= max_steps; n++) {
    auto [e, g] = surface(opt.get_next_geometry());
    opt.update(e, g);
    if (opt.step())
      return n;
  }
  return max_steps;
}

template <typename Opt>
int optimize(Opt &opt, const std::vector<Spring> &springs) {
  return optimize(opt, [&](const occ::core::Molecule &m) {
    return spring_energy(m, springs);
  });
}

double distance(const occ::core::Molecule &m, int i, int j) {
  return (m.positions().col(i) - m.positions().col(j)).norm() *
         occ::units::ANGSTROM_TO_BOHR;
}

occ::core::Molecule distorted_water() {
  occ::IVec z(3);
  z << 8, 1, 1;
  occ::Mat3N pos(3, 3);
  pos << 0.0, 1.1, -0.2, //
      0.0, 0.1, 0.9,     //
      0.0, 0.0, 0.1;
  return occ::core::Molecule(z, pos);
}

const std::vector<Spring> water_springs{
    {0, 1, 0.5, 1.81}, {0, 2, 0.5, 1.81}, {1, 2, 0.1, 2.86}};
} // namespace

TEST_CASE("LbfgsOptimizer reaches the minimum of a model water",
          "[lbfgs_optimizer]") {
  ConvergenceCriteria criteria;
  for (bool precondition : {true, false}) {
    CAPTURE(precondition);
    LbfgsSettings settings;
    settings.precondition = precondition;
    LbfgsOptimizer opt(distorted_water(), criteria, settings);
    const int n = optimize(opt, water_springs);
    REQUIRE(opt.is_converged());
    REQUIRE(n < 50);
    const auto m = opt.get_next_geometry();
    REQUIRE(distance(m, 0, 1) == Catch::Approx(1.81).margin(2e-3));
    REQUIRE(distance(m, 0, 2) == Catch::Approx(1.81).margin(2e-3));
    REQUIRE(distance(m, 1, 2) == Catch::Approx(2.86).margin(2e-3));
  }
}

TEST_CASE("LbfgsOptimizer keeps steps within the trust radius",
          "[lbfgs_optimizer]") {
  LbfgsSettings settings;
  settings.max_step = 0.1;
  settings.precondition = false;
  LbfgsOptimizer opt(distorted_water(), ConvergenceCriteria{}, settings);
  auto before = opt.get_next_geometry();
  for (int i = 0; i < 5; i++) {
    auto [e, g] = spring_energy(before, water_springs);
    opt.update(e, g);
    REQUIRE_FALSE(opt.step());
    auto after = opt.get_next_geometry();
    const double disp =
        ((after.positions() - before.positions()) * occ::units::ANGSTROM_TO_BOHR)
            .colwise()
            .norm()
            .maxCoeff();
    REQUIRE(disp <= opt.trust_radius() * (1 + 1e-12) + 1e-12);
    REQUIRE(disp <= 0.1 + 1e-12);
    before = after;
  }
}

TEST_CASE("LbfgsOptimizer rejects a step that raises the energy",
          "[lbfgs_optimizer]") {
  LbfgsSettings settings;
  settings.precondition = false;
  LbfgsOptimizer opt(distorted_water(), ConvergenceCriteria{}, settings);
  const auto start = opt.get_next_geometry();
  auto [e0, g0] = spring_energy(start, water_springs);
  opt.update(e0, g0);
  REQUIRE_FALSE(opt.step());
  const double trust = opt.trust_radius();
  // Report a higher energy at the new point: the next step starts again from
  // the first point with a smaller trust radius.
  auto [e1, g1] = spring_energy(opt.get_next_geometry(), water_springs);
  opt.update(e0 + 1e-3, g1);
  REQUIRE_FALSE(opt.step());
  REQUIRE(opt.trust_radius() < trust);
  const double from_start =
      ((opt.get_next_geometry().positions() - start.positions()) *
       occ::units::ANGSTROM_TO_BOHR)
          .colwise()
          .norm()
          .maxCoeff();
  REQUIRE(from_start <= opt.trust_radius() * (1 + 1e-12));
}

TEST_CASE("Lindh model Hessian is symmetric, positive semidefinite and has "
          "no translational or rotational stiffness",
          "[model_hessian]") {
  // ethanol, plus acetylene for the linear bends
  occ::IVec z(9);
  z << 6, 6, 8, 1, 1, 1, 1, 1, 1;
  occ::Mat3N pos(3, 9);
  pos << -1.168, 0.255, 0.951, -1.952, -1.271, -1.235, 0.320, 0.312, 1.829,
      -0.408, 0.161, -0.890, 0.315, -0.968, 0.212, 0.787, 0.844, -0.469,
      0.000, 0.000, 0.000, 0.000, 0.893, -0.884, 0.891, -0.876, 0.004;
  occ::IVec z2(4);
  z2 << 1, 6, 6, 1;
  occ::Mat3N pos2(3, 4);
  pos2 << 0, 0, 0, 0.01, 0, 0, 0, 0, -1.66, -0.6, 0.6, 1.66;

  for (const auto &[zs, x] : {std::pair{z, pos}, std::pair{z2, pos2}}) {
    const occ::Mat H = lindh_model_hessian(x, zs);
    const int n = x.cols();
    REQUIRE(H.rows() == 3 * n);
    REQUIRE((H - H.transpose()).norm() < 1e-12);
    Eigen::SelfAdjointEigenSolver<occ::Mat> es(H);
    REQUIRE(es.eigenvalues()(0) > -1e-10);
    REQUIRE(es.eigenvalues().maxCoeff() > 0.1);

    const occ::Mat3N bohr = x * occ::units::ANGSTROM_TO_BOHR;
    const occ::Vec3 com = bohr.rowwise().mean();
    for (int a = 0; a < 3; a++) {
      occ::Vec t = occ::Vec::Zero(3 * n), r = occ::Vec::Zero(3 * n);
      occ::Vec3 axis = occ::Vec3::Zero();
      axis(a) = 1.0;
      for (int i = 0; i < n; i++) {
        t(3 * i + a) = 1.0;
        r.segment<3>(3 * i) = axis.cross(occ::Vec3(bohr.col(i) - com));
      }
      REQUIRE((H * t).norm() < 1e-10);
      REQUIRE((H * r).norm() < 1e-8 * H.norm() * r.norm() + 1e-10);
    }
  }
}

TEST_CASE("Lindh model Hessian gives a rotor about a single bond one "
          "torsion's stiffness, not one per pair of substituents",
          "[model_hessian]") {
  // ethanol: the methyl (C0; H3, H4, H5) turning about C0-C1 changes no bond
  // length or angle, only torsions (and distant pairs)
  occ::IVec z(9);
  z << 6, 6, 8, 1, 1, 1, 1, 1, 1;
  occ::Mat3N pos(3, 9);
  pos << -1.168, 0.255, 0.951, -1.952, -1.271, -1.235, 0.320, 0.312, 1.829,
      -0.408, 0.161, -0.890, 0.315, -0.968, 0.212, 0.787, 0.844, -0.469,
      0.000, 0.000, 0.000, 0.000, 0.893, -0.884, 0.891, -0.876, 0.004;
  const occ::Mat H = lindh_model_hessian(pos, z);
  const occ::Mat3N bohr = pos * occ::units::ANGSTROM_TO_BOHR;
  const occ::Vec3 axis = (bohr.col(1) - bohr.col(0)).normalized();
  occ::Vec v = occ::Vec::Zero(27); // per radian of rotation
  for (int i : {3, 4, 5})
    v.segment<3>(3 * i) = axis.cross(occ::Vec3(bohr.col(i) - bohr.col(0)));
  const double k = v.dot(H * v); // Hartree/rad^2
  INFO("methyl torsion stiffness " << k);
  // about 0.008 from ethanol's 1.2 kcal/mol barrier; the model's terms
  // between non-bonded pairs leave it near 0.04, where the nine summed
  // torsions gave 0.1
  REQUIRE(k > 0.001);
  REQUIRE(k < 0.05);
}

// Springs plus a Lennard-Jones attraction between every pair not joined by a
// spring, so separate fragments bind.
namespace {
std::pair<double, occ::Mat3N> spring_lj_energy(const occ::core::Molecule &m,
                                               const std::vector<Spring> &springs,
                                               double epsilon, double sigma) {
  auto [e, g] = spring_energy(m, springs);
  const occ::Mat3N x = m.positions() * occ::units::ANGSTROM_TO_BOHR;
  for (int i = 0; i < x.cols(); i++) {
    for (int j = i + 1; j < x.cols(); j++) {
      bool bonded = false;
      for (const auto &s : springs)
        bonded |= (s.i == i && s.j == j) || (s.i == j && s.j == i);
      if (bonded)
        continue;
      const occ::Vec3 d = x.col(i) - x.col(j);
      const double r = d.norm(), sr6 = std::pow(sigma / r, 6);
      e += 4 * epsilon * (sr6 * sr6 - sr6);
      const occ::Vec3 f = 4 * epsilon * (-12 * sr6 * sr6 + 6 * sr6) / r * d / r;
      g.col(i) += f;
      g.col(j) -= f;
    }
  }
  return {e, g};
}

} // namespace

TEST_CASE("GeometryOptimizer reaches the minimum of a model water",
          "[geometry_optimizer]") {
  for (auto coords : {OptimizerSettings::Coordinates::TRIC,
                      OptimizerSettings::Coordinates::Cartesian}) {
    OptimizerSettings settings;
    settings.coordinates = coords;
    GeometryOptimizer opt(distorted_water(), ConvergenceCriteria{}, settings);
    const int n = optimize(opt, water_springs);
    CAPTURE(static_cast<int>(coords), n);
    REQUIRE(opt.is_converged());
    REQUIRE(n < 40);
    const auto m = opt.get_next_geometry();
    REQUIRE(distance(m, 0, 1) == Catch::Approx(1.81).margin(2e-3));
    REQUIRE(distance(m, 0, 2) == Catch::Approx(1.81).margin(2e-3));
    REQUIRE(distance(m, 1, 2) == Catch::Approx(2.86).margin(2e-3));
  }
}

TEST_CASE("GeometryOptimizer handles a linear molecule and a bend going linear",
          "[geometry_optimizer]") {
  // three atoms whose minimum is linear (r0 sums: 2.2 + 2.2 = 4.4), started
  // bent at about 150 degrees
  occ::IVec z(3);
  z << 6, 8, 8;
  occ::Mat3N pos(3, 3);
  pos << 0.0, 0.0, 0.3, //
      0.0, 0.0, 0.0,    //
      0.0, 1.2, -1.12;
  const std::vector<Spring> springs{
      {0, 1, 0.8, 2.2}, {0, 2, 0.8, 2.2}, {1, 2, 0.3, 4.4}};
  GeometryOptimizer opt(occ::core::Molecule(z, pos));
  const int n = optimize(opt, springs);
  CAPTURE(n);
  REQUIRE(opt.is_converged());
  const auto m = opt.get_next_geometry();
  REQUIRE(distance(m, 1, 2) == Catch::Approx(4.4).margin(5e-3));
}

TEST_CASE("GeometryOptimizer binds two fragments without intermolecular bonds",
          "[geometry_optimizer]") {
  // two model waters 3.5 Angstrom apart, held by Lennard-Jones
  occ::IVec z(6);
  z << 8, 1, 1, 8, 1, 1;
  occ::Mat3N pos(3, 6);
  pos << 0.0, 0.96, -0.24, 3.5, 4.0, 3.3, //
      0.0, 0.0, 0.93, 0.2, 0.9, -0.8,     //
      0.0, 0.0, 0.0, 0.4, 0.6, 0.5;
  const std::vector<Spring> springs{{0, 1, 0.5, 1.81}, {0, 2, 0.5, 1.81},
                                    {1, 2, 0.1, 2.86}, {3, 4, 0.5, 1.81},
                                    {3, 5, 0.5, 1.81}, {4, 5, 0.1, 2.86}};
  auto energy = [&](const occ::core::Molecule &m) {
    return spring_lj_energy(m, springs, 1e-3, 5.0);
  };
  GeometryOptimizer opt(occ::core::Molecule(z, pos));
  REQUIRE(dynamic_cast<const TRICCoordinates &>(opt.coordinates())
              .num_fragments() == 2);
  const int n = optimize(opt, energy);
  CAPTURE(n);
  REQUIRE(opt.is_converged());
  auto [e, g] = energy(opt.get_next_geometry());
  REQUIRE(g.colwise().norm().maxCoeff() < 3e-4);
}

TEST_CASE("GeometryOptimizer rejects a step that raises the energy",
          "[geometry_optimizer]") {
  GeometryOptimizer opt(distorted_water());
  const auto start = opt.get_next_geometry();
  auto [e0, g0] = spring_energy(start, water_springs);
  opt.update(e0, g0);
  REQUIRE_FALSE(opt.step());
  const double trust = opt.trust_radius();
  auto [e1, g1] = spring_energy(opt.get_next_geometry(), water_springs);
  opt.update(e0 + 1.0, g1);
  REQUIRE_FALSE(opt.step());
  REQUIRE(opt.trust_radius() < trust);
  // the next step starts again from the first geometry
  const occ::Vec d = aligned_displacements(
      start.positions() * occ::units::ANGSTROM_TO_BOHR,
      opt.get_next_geometry().positions() * occ::units::ANGSTROM_TO_BOHR);
  REQUIRE(std::sqrt(d.squaredNorm() / d.size()) <=
          1.1 * opt.trust_radius() + 1e-12);
}

TEST_CASE("A single atom converges at once", "[geometry_optimizer]") {
  occ::IVec z(1);
  z << 2;
  GeometryOptimizer opt(occ::core::Molecule(z, occ::Mat3N::Zero(3, 1)));
  opt.update(-2.9, occ::Mat3N::Zero(3, 1));
  REQUIRE(opt.step());
}

TEST_CASE("Net force and torque projection leaves internal forces",
          "[geometry_optimizer]") {
  const occ::Mat3N x =
      distorted_water().positions() * occ::units::ANGSTROM_TO_BOHR;
  auto [e, g] = spring_energy(distorted_water(), water_springs);
  // the springs' gradient has no net force or torque: unchanged
  REQUIRE((project_net_force_torque(x, g) - g).norm() < 1e-12);
  // a uniform force and a rigid rotation are removed
  occ::Mat3N extra = occ::Mat3N::Zero(3, 3);
  const occ::Vec3 c = x.rowwise().mean();
  for (int i = 0; i < 3; i++)
    extra.col(i) = occ::Vec3(0.1, 0.0, -0.2) +
                   occ::Vec3(0.0, 0.0, 0.3).cross(occ::Vec3(x.col(i) - c));
  REQUIRE((project_net_force_torque(x, g + extra) - g).norm() < 1e-10);
}

// Saddle points: double-well bonds, V = a (r - r1)^2 (r - r2)^2, on top of
// springs, so a bond has a barrier at (r1 + r2) / 2 with curvature
// -a (r2 - r1)^2.
namespace {
struct DoubleWell {
  int i, j;
  double a, r1, r2; // Hartree/Bohr^4, Bohr
};

using Potential = std::function<std::pair<double, occ::Mat3N>(
    const occ::core::Molecule &)>;

Potential double_well_potential(std::vector<Spring> springs,
                                std::vector<DoubleWell> wells) {
  return [=](const occ::core::Molecule &m) {
    auto [e, g] = spring_energy(m, springs);
    const occ::Mat3N x = m.positions() * occ::units::ANGSTROM_TO_BOHR;
    for (const auto &w : wells) {
      const occ::Vec3 d = x.col(w.i) - x.col(w.j);
      const double r = d.norm(), u = r - w.r1, v = r - w.r2;
      e += w.a * u * u * v * v;
      const occ::Vec3 f = 2.0 * w.a * u * v * (u + v) * d / r;
      g.col(w.i) += f;
      g.col(w.j) -= f;
    }
    return std::pair{e, g};
  };
}

// Central differences of the gradient (Hartree/Bohr^2, 3N x 3N).
occ::Mat hessian(const Potential &potential, const occ::core::Molecule &m) {
  const double h = 1e-4; // Bohr
  const int n = m.size();
  occ::Mat H(3 * n, 3 * n);
  for (int c = 0; c < 3 * n; c++) {
    occ::Mat3N x = m.positions();
    x(c % 3, c / 3) += h * occ::units::BOHR_TO_ANGSTROM;
    const occ::Mat3N gp =
        potential(occ::core::Molecule(m.atomic_numbers(), x)).second;
    x(c % 3, c / 3) -= 2.0 * h * occ::units::BOHR_TO_ANGSTROM;
    const occ::Mat3N gm =
        potential(occ::core::Molecule(m.atomic_numbers(), x)).second;
    H.col(c) = (gp - gm).reshaped() / (2.0 * h);
  }
  return 0.5 * (H + H.transpose());
}


// water with O-H bonds (Bohr) r01 and r02 and an H-H spring for the bend
occ::core::Molecule triatomic(double r01, double r02) {
  occ::IVec z(3);
  z << 8, 1, 1;
  const double half = 104.5 / 2.0 * M_PI / 180.0;
  occ::Mat3N pos(3, 3);
  pos.col(0) << 0.0, 0.0, 0.0;
  pos.col(1) << r01 * std::sin(half), 0.0, r01 * std::cos(half);
  pos.col(2) << -r02 * std::sin(half), 0.0, r02 * std::cos(half);
  return occ::core::Molecule(z, pos * occ::units::BOHR_TO_ANGSTROM);
}
} // namespace

TEST_CASE("GeometryOptimizer finds the saddle of a double-well bond",
          "[geometry_optimizer][saddle]") {
  const auto potential = double_well_potential(
      {{0, 2, 0.5, 1.81}, {1, 2, 0.1, 2.86}}, {{0, 1, 1.0, 1.6, 2.4}});
  for (auto coordinates : {OptimizerSettings::Coordinates::TRIC,
                           OptimizerSettings::Coordinates::Cartesian}) {
    CAPTURE(int(coordinates));
    auto settings = OptimizerSettings::saddle_point();
    settings.coordinates = coordinates;
    const auto start = triatomic(1.85, 1.75);
    GeometryOptimizer opt(start, {}, settings);
    opt.set_initial_hessian(hessian(potential, start));
    const int n = optimize(opt, potential);
    INFO("gradients: " << n);
    REQUIRE(opt.is_converged());
    const auto m = opt.get_next_geometry();
    CHECK(distance(m, 0, 1) == Catch::Approx(2.0).margin(2e-3));
    CHECK(distance(m, 0, 2) == Catch::Approx(1.81).margin(2e-3));
    CHECK(opt.negative_modes() == 1);
  }
}

TEST_CASE("GeometryOptimizer follows the requested mode to its saddle",
          "[geometry_optimizer][saddle]") {
  // both bonds have barriers, the first with the more negative curvature:
  // mode 0 leads to the saddle of bond 0-1, mode 1 to that of bond 0-2
  const auto potential = double_well_potential(
      {{1, 2, 0.1, 2.86}},
      {{0, 1, 1.0, 1.6, 2.4}, {0, 2, 0.5, 1.6, 2.4}});
  const auto start = triatomic(1.97, 2.04);
  for (int mode : {0, 1}) {
    CAPTURE(mode);
    GeometryOptimizer opt(start, {}, OptimizerSettings::saddle_point(mode));
    opt.set_initial_hessian(hessian(potential, start));
    const int n = optimize(opt, potential);
    INFO("gradients: " << n);
    REQUIRE(opt.is_converged());
    const auto m = opt.get_next_geometry();
    const double top = distance(m, 0, mode == 0 ? 1 : 2);
    const double other = distance(m, 0, mode == 0 ? 2 : 1);
    CHECK(top == Catch::Approx(2.0).margin(2e-3));
    CHECK(std::min(std::abs(other - 1.6), std::abs(other - 2.4)) < 2e-3);
    CHECK(opt.negative_modes() == 1);
  }
}

// Nudged elastic band
namespace {
int run_neb(NudgedElasticBand &band, const Potential &potential,
            const occ::core::Molecule &reactant,
            const occ::core::Molecule &product, int max_steps = 500) {
  band.set_endpoint_energies(potential(reactant).first,
                             potential(product).first);
  int steps = 0;
  for (; steps < max_steps; steps++) {
    std::vector<double> e;
    std::vector<occ::Mat3N> g;
    for (const auto &image : band.images()) {
      auto [ei, gi] = potential(image);
      e.push_back(ei);
      g.push_back(gi);
    }
    band.update(e, g);
    if (band.step())
      break;
  }
  return steps;
}
} // namespace

TEST_CASE("NEB climbing image finds the saddle of a double-well bond",
          "[neb]") {
  const auto potential = double_well_potential(
      {{0, 2, 0.5, 1.81}, {1, 2, 0.1, 2.86}}, {{0, 1, 1.0, 1.6, 2.4}});
  // endpoints relaxed, as a band needs minima at its ends
  auto relaxed = [&](double r01) {
    GeometryOptimizer opt(triatomic(r01, 1.81));
    optimize(opt, potential);
    return opt.get_next_geometry();
  };
  const auto reactant = relaxed(1.6), product = relaxed(2.4);
  NebSettings settings;
  settings.images = 5;
  NudgedElasticBand band(reactant, product, settings);
  const int steps = run_neb(band, potential, reactant, product);
  INFO("steps: " << steps);
  REQUIRE(band.is_converged());
  REQUIRE(band.climbing());
  const auto path = band.path();
  const int top = band.highest_image();
  CHECK(distance(path[top], 0, 1) == Catch::Approx(2.0).margin(5e-3));
  // the barrier, 1.0 * 0.4^4, over the reactant's energy
  CHECK(band.energies()(top) - band.energies()(0) ==
        Catch::Approx(0.0256).margin(2e-4));
  // the endpoints stay put
  CHECK((path.front().positions() - reactant.positions()).norm() < 1e-8);
}

TEST_CASE("NEB IDPP interpolation keeps atoms apart where a straight line "
          "does not",
          "[neb]") {
  // a hydrogen swinging from one side of an O-C bond to the other, a
  // reflection superposition cannot undo with the second carbon off the
  // plane: the straight line takes it through the oxygen
  occ::IVec z(4);
  z << 8, 1, 6, 6;
  occ::Mat3N r(3, 4);
  r.col(0) << 0.0, 0.0, 0.0;
  r.col(1) << -0.25, 0.93, 0.0;
  r.col(2) << 1.43, 0.0, 0.0;
  r.col(3) << 0.0, 0.0, 1.43;
  occ::Mat3N p = r;
  p.col(1) << -0.25, -0.93, 0.0;
  const occ::core::Molecule reactant(z, r), product(z, p);
  for (bool idpp : {false, true}) {
    CAPTURE(idpp);
    NebSettings settings;
    settings.images = 7;
    settings.idpp = idpp;
    NudgedElasticBand band(reactant, product, settings);
    double closest = 1e10;
    for (const auto &image : band.images())
      closest = std::min(closest, distance(image, 0, 1));
    // Bohr: O-H is 1.82 at the ends; the straight line comes to 1.27
    if (idpp)
      CHECK(closest > 1.7);
    else
      CHECK(closest < 1.4);
  }
}
