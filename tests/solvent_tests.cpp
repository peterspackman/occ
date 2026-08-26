#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/eeq.h>
#include <occ/core/format_matrix.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/io/xyz.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/dft/dft.h>
#include <occ/solvent/sigma_activity.h>
#include <occ/core/element.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/sigma_io.h>
#include <occ/driver/sigma_driver.h>
#include <occ/solvent/sigma_kernel.h>
#include <occ/solvent/sigma_potential.h>
#include <occ/solvent/sigma_profile.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/solvent/surface.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>

using occ::format_matrix;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;
using occ::solvent::COSMO;

// COSMO

TEST_CASE("COSMO self energy", "[solvent]") {

  auto pts = Mat3N(3, 12);
  pts << -0.525731, 0.525731, -0.525731, 0.525731, 0.0, 0.0, 0.0, 0.0, 0.850651,
      0.850651, -0.850651, -0.850651, 0.850651, 0.850651, -0.850651, -0.850651,
      -0.525731, 0.525731, -0.525731, 0.525731, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.850651, 0.850651, -0.850651, -0.850651, -0.525731, 0.525731,
      -0.525731, 0.527531;

  auto areas = Vec(12);
  areas.setConstant(0.79787845);

  auto charges = Vec(12);
  charges.topRows(6).setConstant(-0.05);
  charges.bottomRows(6).setConstant(0.05);
  const COSMO c(78.4);
  COSMO::Result result = c(pts, areas, charges);
  fmt::print("Final energy: {}\n", result.energy);
  fmt::print("InitialFinal charges:\n{}\n", format_matrix(result.initial));
  fmt::print("Converged charges:\n{}\n", format_matrix(result.converged));

  REQUIRE(result.energy == Catch::Approx(-0.0042715403));
}

// SMD tests
const char *WATER = R""""(3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190
)"""";

const char *NAPHTHOL = R""""(19

C          2.27713      0.06621      3.27549
O          4.40801      1.08291      3.23213
H          4.77713      1.76854      2.97536
C          0.99446     -0.07773      2.71091
C          3.16535      0.98455      2.64911
C          2.81588      1.69609      1.55007
H          3.42093      2.28673      1.16155
C          0.65712      0.68419      1.56248
H         -0.18999      0.59447      1.18873
C          1.53795      1.53488      1.00427
H          1.29538      2.01948      0.24814
C          1.71811     -1.55359      4.99111
H          1.95189     -2.04587      5.74511
C          2.61218     -0.69523      4.43172
H          3.45662     -0.60359      4.81044
C          0.44778     -1.69273      4.42995
H         -0.16266     -2.28049      4.81280
C          0.09875     -0.98551      3.34060
H         -0.75476     -1.09298      2.98836
)"""";

const char *AMMONIUM = R""""(5

N    0.0000000   0.0000000   0.0000000
H    0.5889000   0.5889000   0.5889000
H    0.5889000  -0.5889000  -0.5889000
H   -0.5889000   0.5889000  -0.5889000
H   -0.5889000  -0.5889000   0.5889000
)"""";

const char *FORMATE = R""""(4

C    0.0000000   0.0000000   0.0000000
H    0.0000000   0.0000000   1.1200000
O    1.1096000   0.0000000  -0.6400000
O   -1.1096000   0.0000000  -0.6400000
)"""";

const char *UREA = R""""(8

C          0.00000      2.83100      1.55628
H          1.37587      4.20687      1.32520
H          0.80400      3.63500      0.13205
N          0.81136      3.64236      0.87105
O          0.00000      2.83100      2.82017
H         -1.37587      1.45513      1.32520
H         -0.80400      2.02700      0.13205
N         -0.81136      2.01964      0.87105
)"""";

const char *DESLORATADINE = R""""(43

Cl  8.365742  -3.156674   2.279987
N   6.535935   4.241773   2.118667
N   2.124673   2.534338  -0.084953
C   6.154391   3.280493   2.975153
C   7.178945   5.297117   2.634439
C   7.463180   5.440493   3.973800
C   7.079291   4.441900   4.837787
C   6.414559   3.326086   4.351568
C   5.952859   2.186276   5.218267
C   6.759627   0.914848   5.008596
C   6.786974   0.277634   3.624048
C   7.438668  -0.953961   3.539999
C   7.568042  -1.597654   2.328066
C   7.082323  -1.050425   1.154092
C   6.420459   0.151055   1.246275
C   6.218000   0.825342   2.459111
C   5.431605   2.095931   2.432902
C   4.161641   2.158200   1.981929
C   3.356235   0.991035   1.482154
C   2.847468   1.272028   0.051514
C   2.875168   3.659390   0.461818
C   3.381546   3.443666   1.897880
H   1.387702   2.471588   0.307276
H   7.454887   5.987002   2.042481
H   7.915050   6.212564   4.293728
H   7.268853   4.516047   5.765941
H   4.999929   1.998867   5.023960
H   6.019847   2.457190   6.168111
H   6.418401   0.236361   5.643031
H   7.695548   1.108615   5.266166
H   7.796583  -1.352175   4.324456
H   7.197795  -1.527345   0.352463
H   6.083834   0.543509   0.449165
H   2.586004   0.836261   2.084957
H   3.917164   0.175171   1.482154
H   3.621528   1.280187  -0.565749
H   2.251270   0.531511  -0.225938
H   2.297550   4.463256   0.446454
H   3.652100   3.836960  -0.124718
H   3.958665   4.201700   2.165391
H   2.612031   3.407432   2.520566
)"""";

TEST_CASE("SMD CDS energy (naphthol)", "[solvent]") {
  auto mol = occ::io::molecule_from_xyz_string(NAPHTHOL);
  auto nums = mol.atomic_numbers();
  auto pos = mol.positions();
  Mat3N pos_bohr = pos * occ::units::ANGSTROM_TO_BOHR;
  auto params = occ::solvent::get_smd_parameters("water");

  Vec cds_radii = occ::solvent::smd::cds_radii(nums, params);
  auto surface =
      occ::solvent::surface::solvent_surface(cds_radii, nums, pos_bohr, 0.0);
  const auto &surface_positions = surface.vertices;
  const auto &surface_areas = surface.areas;
  const auto &surface_atoms = surface.atom_index;

  auto output = fmt::output_file(
      "desloratadine.cpcm", fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
  output.print("{}\nelement, x, y, z, atom_idx, area\n", surface_areas.rows());
  for (size_t i = 0; i < surface_areas.rows(); i++) {
    output.print("{:4d} {: 12.6f} {: 12.6f} {: 12.6f} {:4d} {: 12.6f}\n",
                 nums(surface_atoms(i)), surface_positions(0, i),
                 surface_positions(1, i), surface_positions(2, i),
                 surface_atoms(i), surface_areas(i));
  }

  Vec surface_areas_per_atom_angs = Vec::Zero(nums.rows());
  const double conversion_factor =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  for (int i = 0; i < surface_areas.rows(); i++) {
    surface_areas_per_atom_angs(surface_atoms(i)) +=
        conversion_factor * surface_areas(i);
  }

  auto surface_tension =
      occ::solvent::smd::atomic_surface_tension(params, nums, pos);

  double H{0.0}, C{0.0}, N{0.0}, Cl{0.0};
  for (int i = 0; i < surface_areas_per_atom_angs.rows(); i++) {
    if (nums(i) == 1)
      H += surface_areas_per_atom_angs(i);
    else if (nums(i) == 6)
      C += surface_areas_per_atom_angs(i);
    else if (nums(i) == 7)
      N += surface_areas_per_atom_angs(i);
    else if (nums(i) == 17)
      Cl += surface_areas_per_atom_angs(i);
    fmt::print("{:<7d} {:10.3f} {:10.3f} {:10.3f}\n", nums(i),
               surface_areas_per_atom_angs(i), surface_tension(i),
               surface_areas_per_atom_angs(i) * surface_tension(i));
  }
  double total_area = surface_areas_per_atom_angs.array().sum();
  fmt::print("Total area = {:.4f}\n", total_area);
  double atomic_term = surface_areas_per_atom_angs.dot(surface_tension);

  fmt::print("CDS energy: {:.4f} kcal/mol\n", atomic_term / 1000);
  fmt::print("SASA:\n");
  fmt::print("H  {:.3f}\n", H);
  fmt::print("C  {:.3f}\n", C);
  fmt::print("N  {:.3f}\n", N);
  fmt::print("Cl {:.3f}\n", Cl);
}

// Cavity construction: probe radius and enclosed volume

namespace {

// Build an N-sphere cavity from explicit radii/centres (Angstrom in, the
// builder works in Bohr).
occ::solvent::surface::Surface
make_cavity(const std::vector<double> &radii_angs,
            const std::vector<occ::Vec3> &centers_angs, double probe_angs) {
  const int n = static_cast<int>(radii_angs.size());
  Vec radii(n);
  occ::IVec nums(n);
  Mat3N pos(3, n);
  for (int i = 0; i < n; i++) {
    radii(i) = radii_angs[i] * occ::units::ANGSTROM_TO_BOHR;
    nums(i) = 6;
    pos.col(i) = centers_angs[i] * occ::units::ANGSTROM_TO_BOHR;
  }
  return occ::solvent::surface::solvent_surface(radii, nums, pos, probe_angs);
}

Mat3N cavity_atom_positions(const std::vector<occ::Vec3> &centers_angs) {
  Mat3N pos(3, centers_angs.size());
  for (size_t i = 0; i < centers_angs.size(); i++)
    pos.col(i) = centers_angs[i] * occ::units::ANGSTROM_TO_BOHR;
  return pos;
}

constexpr double BOHR3_TO_ANGS3 = occ::units::BOHR_TO_ANGSTROM *
                                  occ::units::BOHR_TO_ANGSTROM *
                                  occ::units::BOHR_TO_ANGSTROM;

} // namespace

TEST_CASE("Cavity volume of an isolated sphere", "[solvent][cavity]") {
  // Lebedev weights sum to 1 exactly, so an unmasked sphere recovers
  // V = 1/3 * (4 pi R^2) * R to machine precision.
  const double R = 2.0;
  std::vector<occ::Vec3> centers{occ::Vec3(0.0, 0.0, 0.0)};
  auto surface = make_cavity({R}, centers, 0.0);
  double v = occ::solvent::surface::cavity_volume(
                 surface, cavity_atom_positions(centers)) *
             BOHR3_TO_ANGS3;
  const double expected = 4.0 * occ::units::PI * R * R * R / 3.0;
  REQUIRE(v == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Cavity volume is invariant to placement", "[solvent][cavity]") {
  // The divergence-theorem sum is origin-independent for a closed surface.
  std::vector<double> radii{2.0, 1.8};
  std::vector<occ::Vec3> a{occ::Vec3(0.0, 0.0, 0.0), occ::Vec3(2.6, 0.0, 0.0)};
  std::vector<occ::Vec3> b{occ::Vec3(11.3, -4.1, 7.9),
                           occ::Vec3(13.9, -4.1, 7.9)};

  double va = occ::solvent::surface::cavity_volume(
      make_cavity(radii, a, 0.0), cavity_atom_positions(a));
  double vb = occ::solvent::surface::cavity_volume(
      make_cavity(radii, b, 0.0), cavity_atom_positions(b));
  REQUIRE(va == Catch::Approx(vb).epsilon(1e-8));
}

TEST_CASE("Cavity volume of two overlapping spheres", "[solvent][cavity]") {
  // Union of two equal spheres, analytic:
  //   V = 2*(4/3) pi R^3 - pi (4R + d) (2R - d)^2 / 12
  const double R = 2.0, d = 3.0;
  std::vector<occ::Vec3> centers{occ::Vec3(0.0, 0.0, 0.0),
                                 occ::Vec3(d, 0.0, 0.0)};
  auto surface = make_cavity({R, R}, centers, 0.0);
  double v = occ::solvent::surface::cavity_volume(
                 surface, cavity_atom_positions(centers)) *
             BOHR3_TO_ANGS3;

  const double lens =
      occ::units::PI * (4 * R + d) * (2 * R - d) * (2 * R - d) / 12.0;
  const double expected = 2.0 * 4.0 * occ::units::PI * R * R * R / 3.0 - lens;

  // The boolean mask resolves the intersection circle only to grid
  // resolution (146 Lebedev points per atom), which comes out ~0.5% high.
  REQUIRE(v == Catch::Approx(expected).epsilon(0.01));
}

TEST_CASE("Solvent probe radius is honoured", "[solvent][cavity]") {
  // A larger probe masks more of the crevice between the two spheres.
  const double R = 2.0, d = 3.0;
  std::vector<occ::Vec3> centers{occ::Vec3(0.0, 0.0, 0.0),
                                 occ::Vec3(d, 0.0, 0.0)};
  auto bare = make_cavity({R, R}, centers, 0.0);
  auto probed = make_cavity({R, R}, centers, 1.3);

  REQUIRE(probed.areas.size() < bare.areas.size());
  REQUIRE(probed.areas.sum() < bare.areas.sum());

  // Surviving points are projected back to the atom radius in both cases.
  Mat3N pos = cavity_atom_positions(centers);
  const double R_bohr = R * occ::units::ANGSTROM_TO_BOHR;
  for (const auto *s : {&bare, &probed}) {
    for (Eigen::Index i = 0; i < s->areas.size(); i++) {
      double r = (s->vertices.col(i) - pos.col(s->atom_index(i))).norm();
      REQUIRE(r == Catch::Approx(R_bohr).epsilon(1e-9));
    }
  }
}

TEST_CASE("Zero probe reproduces the legacy cavity", "[solvent][cavity]") {
  // A 0.001 A probe is indistinguishable from none, which is what keeps
  // the SMD and CPCM-X cavities fixed.
  std::vector<double> radii{2.0, 1.8, 1.3};
  std::vector<occ::Vec3> centers{occ::Vec3(0.0, 0.0, 0.0),
                                 occ::Vec3(2.6, 0.0, 0.0),
                                 occ::Vec3(-1.0, 1.0, 0.4)};
  auto zero = make_cavity(radii, centers, 0.0);
  auto legacy = make_cavity(radii, centers, 0.001);
  REQUIRE(zero.areas.size() == legacy.areas.size());
  REQUIRE(zero.areas.sum() == Catch::Approx(legacy.areas.sum()).epsilon(1e-5));
}

// Sigma profiles

namespace {

using occ::solvent::sigma::Grid;
using occ::solvent::sigma::HBondClass;
using occ::solvent::sigma::Segments;

// Segments laid out along x, in Angstrom, with uniform area.
Segments line_segments(const std::vector<double> &x_angs,
                       const std::vector<double> &sigmas, double area_angs2) {
  const int n = static_cast<int>(x_angs.size());
  Segments s;
  s.positions = Mat3N::Zero(3, n);
  s.areas = Vec::Constant(n, area_angs2);
  s.sigma = Vec(n);
  s.atom_index = occ::IVec::Zero(n);
  s.atomic_number = occ::IVec::Constant(n, 6);
  s.hbond_class = occ::IVec::Zero(n);
  for (int i = 0; i < n; i++) {
    s.positions(0, i) = x_angs[i] * occ::units::ANGSTROM_TO_BOHR;
    s.sigma(i) = sigmas[i];
  }
  return s;
}

double p_hb(double sigma, double sigma_0 = 0.007) {
  return 1.0 - std::exp(-sigma * sigma / (2.0 * sigma_0 * sigma_0));
}

} // namespace

TEST_CASE("Sigma grid nodes", "[solvent][sigma]") {
  Grid g;
  REQUIRE(g.n == 51);
  REQUIRE(g.spacing() == Catch::Approx(0.001));
  Vec c = g.centers();
  REQUIRE(c(0) == Catch::Approx(-0.025));
  REQUIRE(c(25) == Catch::Approx(0.0).margin(1e-15));
  REQUIRE(c(50) == Catch::Approx(0.025));
}

TEST_CASE("Segments carry charge density, not charge", "[solvent][sigma]") {
  // The COSMO solver returns segment charges; sigma is that over the area.
  occ::solvent::surface::Surface cavity;
  cavity.vertices = Mat3N::Zero(3, 2);
  cavity.vertices(0, 1) = 2.0;
  cavity.areas = Vec(2);
  cavity.areas << 1.0, 4.0; // Bohr^2
  cavity.atom_index = occ::IVec::Zero(2);

  Vec charges(2);
  charges << -0.3, 0.1;

  occ::IVec nums(1);
  nums << 8;
  auto s = occ::solvent::sigma::segments_from_cavity(cavity, charges, nums);
  const double conv =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  REQUIRE(s.areas(0) == Catch::Approx(conv));
  REQUIRE(s.sigma(0) == Catch::Approx(-0.3 / conv));
  REQUIRE(s.sigma(1) == Catch::Approx(0.1 / (4.0 * conv)));
  // Total screening charge is recovered exactly.
  REQUIRE(s.total_charge() == Catch::Approx(-0.2));
}

TEST_CASE("Segment averaging is a local weighted mean", "[solvent][sigma]") {
  auto s = line_segments({0.0, 0.05, 10.0}, {0.01, -0.01, 0.05}, 0.35);
  occ::solvent::sigma::average_sigma(s, 0.5);

  // Two near-coincident segments average to each other.
  REQUIRE(s.sigma_averaged(0) == Catch::Approx(s.sigma_averaged(1)).margin(1e-3));
  REQUIRE(std::abs(s.sigma_averaged(0)) < 2e-3);
  // An isolated segment keeps its own value.
  REQUIRE(s.sigma_averaged(2) == Catch::Approx(0.05).epsilon(1e-6));
  // A weighted mean never leaves the range of its inputs.
  REQUIRE(s.sigma_averaged.minCoeff() >= s.sigma.minCoeff() - 1e-12);
  REQUIRE(s.sigma_averaged.maxCoeff() <= s.sigma.maxCoeff() + 1e-12);
}

TEST_CASE("Averaging a constant field is the identity", "[solvent][sigma]") {
  auto s = line_segments({0.0, 0.3, 0.6, 0.9}, {0.007, 0.007, 0.007, 0.007},
                         0.35);
  occ::solvent::sigma::average_sigma(s, 0.5);
  for (Eigen::Index i = 0; i < s.size(); i++)
    REQUIRE(s.sigma_averaged(i) == Catch::Approx(0.007).epsilon(1e-12));
}

TEST_CASE("Binning conserves area", "[solvent][sigma]") {
  auto s = line_segments({0.0, 0.3, 0.6}, {-0.0123, 0.0004, 0.0176}, 0.4);
  s.sigma_averaged = s.sigma;
  Grid g;
  double outside = 0.0;
  auto p = occ::solvent::sigma::bin_segments(s, g, {}, &outside);
  REQUIRE(p.total_area() == Catch::Approx(s.total_area()).epsilon(1e-14));
  REQUIRE(outside == Catch::Approx(0.0).margin(1e-15));
  REQUIRE(p.num_classes() == 1);
  REQUIRE(p.normalized().sum() == Catch::Approx(1.0));
}

TEST_CASE("Binning clamps and reports out-of-range area", "[solvent][sigma]") {
  auto s = line_segments({0.0, 0.3}, {-0.04, 0.002}, 0.4);
  s.sigma_averaged = s.sigma;
  Grid g;
  double outside = 0.0;
  auto p = occ::solvent::sigma::bin_segments(s, g, {}, &outside);
  REQUIRE(p.total_area() == Catch::Approx(s.total_area()).epsilon(1e-14));
  REQUIRE(outside == Catch::Approx(0.4));
}

TEST_CASE("Binning and segment contraction are adjoint", "[solvent][sigma]") {
  // Depositing area onto the grid and interpolating a field off it use the
  // same weights, so the binned and segment-resolved contractions agree
  // exactly. This is what lets a per-patch attribution skip re-binning.
  auto s = line_segments({0.0, 0.3, 0.6, 0.9, 1.2},
                         {-0.0181, -0.0034, 0.0002, 0.0091, 0.0203}, 0.37);
  s.sigma_averaged = s.sigma;
  Grid g;

  Mat field(g.n, 1);
  Vec centers = g.centers();
  for (int i = 0; i < g.n; i++)
    field(i, 0) = 3.1 - 240.0 * centers(i) + 1.5e4 * centers(i) * centers(i);

  auto p = occ::solvent::sigma::bin_segments(s, g);
  double binned = occ::solvent::sigma::contract(p, field);
  Vec per_segment = occ::solvent::sigma::contract_segments(s, g, field);

  REQUIRE(per_segment.size() == s.size());
  REQUIRE(per_segment.sum() == Catch::Approx(binned).epsilon(1e-13));
}

TEST_CASE("H-bond classification of segments", "[solvent][sigma]") {
  // Water: the oxygen carries hydrogens, so it and they are the OH class.
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  auto s = line_segments({0.0, 0.3, 0.6}, {0.0, 0.0, 0.0}, 0.4);
  s.atom_index << 0, 1, 2; // O, H, H

  Mat3N pos_bohr = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
  occ::solvent::sigma::classify_hbond_segments(s, mol.atomic_numbers(),
                                               pos_bohr);
  for (int i = 0; i < 3; i++)
    REQUIRE(s.hbond_class(i) == static_cast<int>(HBondClass::OH));
}

TEST_CASE("Carbonyl and amine atoms are the OT class", "[solvent][sigma]") {
  // Urea: C=O has no attached hydrogen and the N-H hydrogens sit on nitrogen,
  // so none of it is the hydroxyl class.
  auto mol = occ::io::molecule_from_xyz_string(UREA);
  auto nums = mol.atomic_numbers();
  const int natoms = static_cast<int>(nums.size());

  Segments s = line_segments(std::vector<double>(natoms, 0.0),
                             std::vector<double>(natoms, 0.0), 0.4);
  for (int i = 0; i < natoms; i++)
    s.atom_index(i) = i;

  Mat3N pos_bohr = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
  occ::solvent::sigma::classify_hbond_segments(s, nums, pos_bohr);

  for (int i = 0; i < natoms; i++) {
    const int expected = (nums(i) == 6) ? static_cast<int>(HBondClass::None)
                                        : static_cast<int>(HBondClass::OT);
    REQUIRE(s.hbond_class(i) == expected);
  }
}

TEST_CASE("H-bond profile split is fractional in sigma",
          "[solvent][sigma]") {
  // Acceptor lobe on O, donor lobe on H, and a non-bonding C segment. Each
  // H-bonding segment keeps only P_hb(sigma) of its area in its own column.
  auto s = line_segments({0.0, 3.0, 6.0}, {0.015, -0.015, 0.001}, 0.4);
  s.sigma_averaged = s.sigma;
  s.atomic_number << 8, 1, 6;
  s.hbond_class << static_cast<int>(HBondClass::OH),
      static_cast<int>(HBondClass::OH), static_cast<int>(HBondClass::None);

  Grid g;
  occ::solvent::sigma::HBondSplit split{true, 0.007};
  auto profile = occ::solvent::sigma::bin_segments(s, g, split);

  REQUIRE(profile.num_classes() == 3);
  REQUIRE(profile.total_area() == Catch::Approx(1.2).epsilon(1e-14));

  const int nhb = static_cast<int>(HBondClass::None);
  const int oh = static_cast<int>(HBondClass::OH);
  const double w = p_hb(0.015);
  // sigma = -0.015, 0.001 and 0.015 land exactly on nodes 10, 26 and 40.
  REQUIRE(profile.values(40, oh) == Catch::Approx(0.4 * w));
  REQUIRE(profile.values(10, oh) == Catch::Approx(0.4 * w));
  REQUIRE(profile.values(40, nhb) == Catch::Approx(0.4 * (1.0 - w)));
  REQUIRE(profile.values(26, nhb) == Catch::Approx(0.4));
  REQUIRE(profile.values.col(static_cast<int>(HBondClass::OT)).sum() ==
          Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("Only the H-bonding lobe of an atom participates",
          "[solvent][sigma]") {
  // A hydroxyl oxygen carrying negative sigma is the donor side of the atom,
  // not the acceptor side, so none of it is available to hydrogen bond.
  auto s = line_segments({0.0}, {-0.015}, 0.4);
  s.sigma_averaged = s.sigma;
  s.atomic_number << 8;
  s.hbond_class << static_cast<int>(HBondClass::OH);

  Grid g;
  auto profile =
      occ::solvent::sigma::bin_segments(s, g, {true, 0.007});
  REQUIRE(profile.values(10, static_cast<int>(HBondClass::None)) ==
          Catch::Approx(0.4));
  REQUIRE(profile.values.col(static_cast<int>(HBondClass::OH)).sum() ==
          Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("Split binning and contraction stay adjoint", "[solvent][sigma]") {
  auto s = line_segments({0.0, 3.0, 6.0, 9.0}, {0.019, -0.012, 0.0035, -0.021},
                         0.31);
  s.sigma_averaged = s.sigma;
  s.atomic_number << 8, 1, 7, 1;
  s.hbond_class << static_cast<int>(HBondClass::OH),
      static_cast<int>(HBondClass::OH), static_cast<int>(HBondClass::OT),
      static_cast<int>(HBondClass::OT);

  Grid g;
  occ::solvent::sigma::HBondSplit split{true, 0.007};
  Mat field(g.n, occ::solvent::sigma::num_hbond_classes);
  Vec centers = g.centers();
  for (int i = 0; i < g.n; i++) {
    field(i, 0) = 2.0 - 130.0 * centers(i);
    field(i, 1) = -8.0 + 400.0 * centers(i) * centers(i);
    field(i, 2) = 1.3 + 55.0 * centers(i);
  }

  auto profile = occ::solvent::sigma::bin_segments(s, g, split);
  double binned = occ::solvent::sigma::contract(profile, field);
  Vec per_segment =
      occ::solvent::sigma::contract_segments(s, g, field, split);
  REQUIRE(per_segment.sum() == Catch::Approx(binned).epsilon(1e-13));
}

TEST_CASE("Mixture profiles weight by mole fraction", "[solvent][sigma]") {
  auto a = line_segments({0.0}, {-0.010}, 1.0);
  auto b = line_segments({0.0}, {0.010}, 3.0);
  a.sigma_averaged = a.sigma;
  b.sigma_averaged = b.sigma;
  Grid g;
  std::vector<occ::solvent::sigma::Profile> components{
      occ::solvent::sigma::bin_segments(a, g),
      occ::solvent::sigma::bin_segments(b, g)};

  Vec x(2);
  x << 0.25, 0.75;
  auto mixture = occ::solvent::sigma::mix_profiles(components, x);
  REQUIRE(mixture.total_area() == Catch::Approx(0.25 * 1.0 + 0.75 * 3.0));
  REQUIRE(mixture.values(15, 0) == Catch::Approx(0.25 * 1.0));
  REQUIRE(mixture.values(35, 0) == Catch::Approx(0.75 * 3.0));
}

// Sigma kernel and potential

namespace {

using occ::solvent::sigma::Kernel;
using occ::solvent::sigma::Model;
using occ::solvent::sigma::Parameters;
using occ::solvent::sigma::Potential;
using occ::solvent::sigma::PotentialOptions;
using occ::solvent::sigma::Profile;

// Bimodal, water-like: a donor lobe and an acceptor lobe.
Profile synthetic_profile(const Grid &g, int num_classes) {
  Profile p;
  p.grid = g;
  p.values = Mat::Zero(g.n, num_classes);
  Vec centers = g.centers();
  for (int i = 0; i < g.n; i++) {
    const double x = centers(i);
    const double lobes = std::exp(-std::pow((x - 0.013) / 0.005, 2)) +
                         std::exp(-std::pow((x + 0.013) / 0.005, 2)) +
                         0.3 * std::exp(-std::pow(x / 0.004, 2));
    p.values(i, 0) = 0.4 * lobes;
    if (num_classes > 1) {
      p.values(i, 1) = 0.35 * std::exp(-std::pow((x - 0.015) / 0.004, 2));
      p.values(i, 2) = 0.2 * std::exp(-std::pow((x + 0.015) / 0.004, 2));
    }
  }
  return p;
}

// A kernel with prescribed matrices, for the analytic fixed points.
Kernel constant_kernel(const Grid &g, double value) {
  Kernel k;
  k.grid = g;
  k.num_classes = 1;
  k.misfit = Mat::Constant(g.n, g.n, value);
  k.hbond = Mat::Zero(g.n, g.n);
  return k;
}

} // namespace

TEST_CASE("Sigma kernel is symmetric", "[solvent][sigma][kernel]") {
  Grid g;
  for (auto model : {Model::CosmoSac2002, Model::CosmoSac2010}) {
    auto params = Parameters::for_model(model);
    auto k = occ::solvent::sigma::build_kernel(g, params, 298.15);
    REQUIRE((k.misfit - k.misfit.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE((k.hbond - k.hbond.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    REQUIRE(k.dim() == g.n * params.num_classes());
  }
}

TEST_CASE("Electrostatic misfit coefficient", "[solvent][sigma][kernel]") {
  auto p10 = Parameters::cosmo_sac_2010();
  REQUIRE(p10.c_es(298.15) ==
          Catch::Approx(6525.69 + 1.4859e8 / (298.15 * 298.15)));
  // The 2002 misfit is temperature independent.
  auto p02 = Parameters::cosmo_sac_2002();
  REQUIRE(p02.c_es(250.0) == Catch::Approx(16466.72 / 2.0));
  REQUIRE(p02.c_es(400.0) == Catch::Approx(16466.72 / 2.0));
}

TEST_CASE("COSMO-SAC 2002 hydrogen bonding is a threshold term",
          "[solvent][sigma][kernel]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2002();
  auto k = occ::solvent::sigma::build_kernel(g, params, 298.15);
  Vec centers = g.centers();

  for (int a = 0; a < g.n; a++) {
    for (int b = 0; b < g.n; b++) {
      const double acc = std::max(centers(a), centers(b));
      const double don = std::min(centers(a), centers(b));
      const bool active = acc > params.sigma_hb && don < -params.sigma_hb;
      if (!active)
        REQUIRE(k.hbond(a, b) == Catch::Approx(0.0).margin(1e-14));
      else
        REQUIRE(k.hbond(a, b) < 0.0);
    }
  }
}

TEST_CASE("COSMO-SAC 2010 hydrogen bonding is gated on sign and class",
          "[solvent][sigma][kernel]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto k = occ::solvent::sigma::build_kernel(g, params, 298.15);
  Vec centers = g.centers();
  const int nhb = static_cast<int>(HBondClass::None);
  const int oh = static_cast<int>(HBondClass::OH);
  const int ot = static_cast<int>(HBondClass::OT);
  auto index = [&](int cls, int bin) { return cls * g.n + bin; };

  // Nodes 40 and 10 are sigma = +0.015 and -0.015.
  const int hi = 40, lo = 10;
  const double diff = centers(hi) - centers(lo);

  REQUIRE(k.hbond(index(oh, hi), index(oh, lo)) ==
          Catch::Approx(-params.c_oh_oh * diff * diff));
  REQUIRE(k.hbond(index(ot, hi), index(ot, lo)) ==
          Catch::Approx(-params.c_ot_ot * diff * diff));
  REQUIRE(k.hbond(index(oh, hi), index(ot, lo)) ==
          Catch::Approx(-params.c_oh_ot * diff * diff));

  // Same sign, or a non-bonding partner, switches the term off entirely.
  REQUIRE(k.hbond(index(oh, hi), index(oh, hi)) ==
          Catch::Approx(0.0).margin(1e-14));
  REQUIRE(k.hbond(index(oh, hi), index(nhb, lo)) ==
          Catch::Approx(0.0).margin(1e-14));

  // The misfit term does not depend on the class.
  REQUIRE(k.misfit(index(oh, hi), index(ot, lo)) ==
          Catch::Approx(k.misfit(index(nhb, hi), index(nhb, lo))));
}

TEST_CASE("Pairing distribution is normalised", "[solvent][sigma][potential]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto profile = synthetic_profile(g, params.num_classes());
  auto potential = occ::solvent::sigma::solve_sigma_potential(profile, kernel);

  Vec flat = occ::solvent::sigma::flatten(profile.normalized());
  Mat pairing = occ::solvent::sigma::pairing_matrix(
      flat, kernel, occ::solvent::sigma::flatten(potential.mu), 298.15);

  REQUIRE(pairing.rows() == kernel.dim());
  for (Eigen::Index i = 0; i < pairing.rows(); i++)
    REQUIRE(pairing.row(i).sum() == Catch::Approx(1.0).epsilon(1e-13));
  REQUIRE(pairing.minCoeff() >= 0.0);
}

TEST_CASE("Analytic fixed point for a constant kernel",
          "[solvent][sigma][potential]") {
  // With E(s,s') = c the closure collapses to mu = -mu + c, so mu = c/2
  // everywhere and the pairing energy has no spread at all.
  Grid g{5, -0.02, 0.02};
  const double c = 1.7;
  auto kernel = constant_kernel(g, c);
  auto profile = synthetic_profile(g, 1);
  auto potential = occ::solvent::sigma::solve_sigma_potential(profile, kernel);

  REQUIRE(potential.converged);
  for (int i = 0; i < g.n; i++) {
    REQUIRE(potential.mu(i, 0) == Catch::Approx(c / 2.0).epsilon(1e-10));
    REQUIRE(potential.mean_energy(i, 0) == Catch::Approx(c).epsilon(1e-10));
    REQUIRE(potential.variance(i, 0) == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("Zero kernel gives a vanishing potential",
          "[solvent][sigma][potential]") {
  Grid g{5, -0.02, 0.02};
  auto kernel = constant_kernel(g, 0.0);
  auto profile = synthetic_profile(g, 1);
  auto potential = occ::solvent::sigma::solve_sigma_potential(profile, kernel);
  REQUIRE(potential.converged);
  REQUIRE(potential.mu.cwiseAbs().maxCoeff() < 1e-12);
  REQUIRE(potential.pairing_entropy.cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Newton and damped iteration agree", "[solvent][sigma][potential]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto profile = synthetic_profile(g, params.num_classes());

  PotentialOptions newton;
  newton.use_newton = true;
  auto a = occ::solvent::sigma::solve_sigma_potential(profile, kernel, newton);

  REQUIRE(a.converged);
  for (double mixing : {0.2, 0.3, 0.5}) {
    PotentialOptions picard;
    picard.use_newton = false;
    picard.mixing = mixing;
    picard.max_iterations = 20000;
    auto b =
        occ::solvent::sigma::solve_sigma_potential(profile, kernel, picard);
    REQUIRE(b.converged);
    REQUIRE((a.mu - b.mu).cwiseAbs().maxCoeff() < 1e-8);
    REQUIRE((a.variance - b.variance).cwiseAbs().maxCoeff() < 1e-6);
  }
  // Newton should get there in far fewer steps than damped substitution.
  REQUIRE(a.iterations < 30);
}

TEST_CASE("Converged potential satisfies its own closure",
          "[solvent][sigma][potential]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto profile = synthetic_profile(g, params.num_classes());
  auto potential = occ::solvent::sigma::solve_sigma_potential(profile, kernel);
  REQUIRE(potential.converged);

  // Substitute the answer back into mu = -(1/beta) ln Z.
  const double rt = occ::solvent::sigma::gas_constant_kcal * 298.15;
  Vec flat_p = occ::solvent::sigma::flatten(profile.normalized());
  Vec mu = occ::solvent::sigma::flatten(potential.mu);
  Mat energy = kernel.total();

  for (Eigen::Index i = 0; i < kernel.dim(); i++) {
    double z = 0.0;
    for (Eigen::Index j = 0; j < kernel.dim(); j++)
      z += flat_p(j) * std::exp((mu(j) - energy(i, j)) / rt);
    REQUIRE(-rt * std::log(z) == Catch::Approx(mu(i)).epsilon(1e-9));
  }
}

TEST_CASE("Variance splits exactly into its terms",
          "[solvent][sigma][potential]") {
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto profile = synthetic_profile(g, params.num_classes());
  auto p = occ::solvent::sigma::solve_sigma_potential(profile, kernel);

  Mat reconstructed =
      p.variance_misfit + p.variance_hbond + 2.0 * p.covariance;
  const double scale = p.variance.cwiseAbs().maxCoeff();
  REQUIRE((p.variance - reconstructed).cwiseAbs().maxCoeff() < 1e-12 * scale);

  REQUIRE(p.variance.minCoeff() >= -1e-12 * scale);
  REQUIRE(p.variance_misfit.minCoeff() >= -1e-12 * scale);
  REQUIRE(p.variance_hbond.minCoeff() >= -1e-12 * scale);
  REQUIRE(p.hbond_probability.minCoeff() >= 0.0);
  REQUIRE(p.hbond_probability.maxCoeff() <= 1.0);
}

TEST_CASE("Convergence is gated on the variance too",
          "[solvent][sigma][potential]") {
  // The second moment settles later than the potential, so tightening only
  // the variance tolerance must make the solver work harder.
  Grid g;
  auto params = Parameters::cosmo_sac_2010();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto profile = synthetic_profile(g, params.num_classes());

  PotentialOptions loose;
  loose.use_newton = false;
  loose.tolerance_mu = 1e-6;
  loose.tolerance_variance = 1e30;
  loose.max_iterations = 20000;
  auto a = occ::solvent::sigma::solve_sigma_potential(profile, kernel, loose);

  PotentialOptions tight = loose;
  tight.tolerance_variance = 1e-10;
  auto b = occ::solvent::sigma::solve_sigma_potential(profile, kernel, tight);

  REQUIRE(a.converged);
  REQUIRE(b.converged);
  REQUIRE(b.iterations > a.iterations);

  auto reference =
      occ::solvent::sigma::solve_sigma_potential(profile, kernel, {});
  const double lag = (a.variance - reference.variance).cwiseAbs().maxCoeff();
  fmt::print("variance still moving at tol_mu=1e-6: {:.3e} (kcal/mol)^2\n", lag);
  REQUIRE(lag > loose.tolerance_mu);
}

TEST_CASE("The potential is defined where the solvent has no area",
          "[solvent][sigma][potential]") {
  // A solute segment can land in a bin the solvent never occupies, so mu has
  // to be finite across the whole grid.
  Grid g;
  auto params = Parameters::cosmo_sac_2002();
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);

  Profile sparse;
  sparse.grid = g;
  sparse.values = Mat::Zero(g.n, 1);
  sparse.values(20, 0) = 1.0;
  sparse.values(30, 0) = 2.0;

  auto p = occ::solvent::sigma::solve_sigma_potential(sparse, kernel);
  REQUIRE(p.converged);
  REQUIRE(p.mu.allFinite());
  REQUIRE(p.variance.allFinite());
}

TEST_CASE("Ideal-conductor COSMO profile for water",
          "[solvent][sigma][conductor]") {
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  auto atoms = mol.atoms();

  occ::gto::AOBasis basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(true);
  occ::dft::DFT gas_ks("b3lyp", basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
  double gas_energy = gas_scf.compute_scf_energy();
  auto gas_wfn = gas_scf.wavefunction();

  occ::driver::SigmaProfileSettings settings;
  settings.basis = "def2-svp";
  auto result = occ::driver::conductor_profile(gas_wfn, settings);

  // A conductor is the strongest possible dielectric, so it must stabilise.
  REQUIRE(result.energy_conductor < gas_energy);

  const auto &segments = result.segments;
  REQUIRE(segments.size() > 100);
  REQUIRE(segments.sigma_averaged.size() == segments.size());

  // Klamt radii on water give a cavity of roughly 43 A^2 and 25 A^3.
  REQUIRE(result.cavity_area > 30.0);
  REQUIRE(result.cavity_area < 60.0);
  REQUIRE(result.cavity_volume > 15.0);
  REQUIRE(result.cavity_volume < 45.0);

  // Neutral solute, so the screening charge integrates to zero up to the
  // cavity discretisation error.
  fmt::print("water conductor: {} segments, area {:.2f} A^2, volume {:.2f} "
             "A^3, screening charge {:+.5f} e\n",
             segments.size(), result.cavity_area, result.cavity_volume,
             result.screening_charge);
  REQUIRE(std::abs(result.screening_charge) < 0.05);

  // Screening charge is opposite in sign to the solute charge it screens:
  // positive over the electronegative oxygen, negative over the hydrogens.
  double o_sigma = 0.0, o_area = 0.0, h_sigma = 0.0, h_area = 0.0;
  for (Eigen::Index i = 0; i < segments.size(); i++) {
    if (segments.atomic_number(i) == 8) {
      o_sigma += segments.areas(i) * segments.sigma_averaged(i);
      o_area += segments.areas(i);
    } else {
      h_sigma += segments.areas(i) * segments.sigma_averaged(i);
      h_area += segments.areas(i);
    }
  }
  fmt::print("mean sigma: O {:+.5f}, H {:+.5f} e/A^2\n", o_sigma / o_area,
             h_sigma / h_area);
  REQUIRE(o_sigma / o_area > 0.0);
  REQUIRE(h_sigma / h_area < 0.0);

  // Water is all hydroxyl, so every segment is the OH class.
  for (Eigen::Index i = 0; i < segments.size(); i++)
    REQUIRE(segments.hbond_class(i) == static_cast<int>(HBondClass::OH));
}

TEST_CASE("Water sigma potential", "[solvent][sigma][conductor]") {
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
  basis.set_pure(true);
  occ::dft::DFT gas_ks("b3lyp", basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
  gas_scf.compute_scf_energy();

  occ::driver::SigmaProfileSettings settings;
  settings.basis = "def2-svp";
  auto conductor =
      occ::driver::conductor_profile(gas_scf.wavefunction(), settings);

  auto params = Parameters::cosmo_sac_2010();
  Grid g;
  auto profile = occ::solvent::sigma::bin_segments(conductor.segments, g,
                                                   params.hbond_split());
  auto kernel = occ::solvent::sigma::build_kernel(g, params, 298.15);
  auto potential = occ::solvent::sigma::solve_sigma_potential(profile, kernel);
  REQUIRE(potential.converged);

  Vec centers = g.centers();
  fmt::print("\n   sigma     p_nhb    p_oh     mu_nhb    mu_oh     var_oh   pHB_oh\n");
  for (int i = 0; i < g.n; i += 5) {
    fmt::print("{:+8.4f} {:8.4f} {:8.4f} {:+9.3f} {:+9.3f} {:9.3f} {:7.3f}\n",
               centers(i), profile.values(i, 0), profile.values(i, 1),
               potential.mu(i, 0), potential.mu(i, 1), potential.variance(i, 1),
               potential.hbond_probability(i, 1));
  }

  const int oh = static_cast<int>(HBondClass::OH);
  const int nhb = static_cast<int>(HBondClass::None);
  // Nodes 5, 10, 25, 40 and 45 are sigma = -0.020, -0.015, 0, +0.015, +0.020.
  const int lo = 5, donor = 10, mid = 25, acceptor = 40, hi = 45;

  // The profile is bimodal, with the hydroxyl lobes at roughly +/-0.015.
  Eigen::Index peak_negative = 0, peak_positive = 0;
  profile.values.col(oh).head(g.n / 2).maxCoeff(&peak_negative);
  profile.values.col(oh).tail(g.n / 2).maxCoeff(&peak_positive);
  peak_positive += g.n / 2;
  REQUIRE(centers(peak_negative) < -0.010);
  REQUIRE(centers(peak_positive) > 0.010);

  // Water donates and accepts, so a hydrogen-bonding solute segment is
  // stabilised at both extremes and penalised in the non-polar middle.
  REQUIRE(potential.mu(lo, oh) < 0.0);
  REQUIRE(potential.mu(hi, oh) < 0.0);
  REQUIRE(potential.mu(mid, oh) > 0.0);
  REQUIRE(potential.mu(lo, oh) < potential.mu(mid, oh));
  REQUIRE(potential.mu(hi, oh) < potential.mu(mid, oh));

  // A non-bonding segment has no such channel: the 2010 electrostatic term
  // is purely a misfit penalty, so it is uphill at both extremes.
  REQUIRE(potential.mu(lo, nhb) > 0.0);
  REQUIRE(potential.mu(hi, nhb) > 0.0);

  // The H-bond channel is two-state: saturated in the wings, off in the
  // middle.
  REQUIRE(potential.hbond_probability(lo, oh) > 0.95);
  REQUIRE(potential.hbond_probability(hi, oh) > 0.95);
  REQUIRE(potential.hbond_probability(mid, oh) < 0.5);

  // The pairing energy spreads out where that channel is genuinely mixed and
  // collapses where it saturates.
  REQUIRE(potential.variance(mid, oh) < potential.variance(donor, oh));
  REQUIRE(potential.variance(mid, oh) < potential.variance(acceptor, oh));
}

// Numerical validation against the NIST reference implementation
// (usnistgov/COSMOSAC). tests/data/cosmo_sac_reference.json carries the sigma
// profiles it was driven with and the ln(gamma) it produced, so both codes
// see byte-identical input and any difference is ours.

#ifndef OCC_TEST_DATA_DIR
#define OCC_TEST_DATA_DIR "data"
#endif

namespace {

using occ::solvent::sigma::Component;

struct ReferenceCase {
  std::vector<std::string> components;
  double temperature;
  Vec mole_fractions;
  Vec ln_gamma_residual;
  Vec ln_gamma_combinatorial;
};

struct ReferenceSet {
  ankerl::unordered_dense::map<std::string, Component> components;
  std::vector<ReferenceCase> cases;
};

ReferenceSet load_reference(const std::string &model_key) {
  std::ifstream input(std::string(OCC_TEST_DATA_DIR) +
                      "/cosmo_sac_reference.json");
  REQUIRE(input.good());
  auto json = nlohmann::json::parse(input);
  const auto &block = json.at(model_key);

  ReferenceSet set;
  for (const auto &[name, entry] : block.at("components").items()) {
    auto sigma = entry.at("sigma").get<std::vector<double>>();
    auto columns =
        entry.at("psigmaA").get<std::vector<std::vector<double>>>();

    Component component;
    component.volume = entry.at("volume").get<double>();
    component.profile.grid =
        Grid{static_cast<int>(sigma.size()), sigma.front(), sigma.back()};
    component.profile.values =
        Mat(sigma.size(), static_cast<Eigen::Index>(columns.size()));
    for (size_t c = 0; c < columns.size(); c++)
      for (size_t i = 0; i < sigma.size(); i++)
        component.profile.values(i, c) = columns[c][i];
    set.components.emplace(name, std::move(component));
  }

  for (const auto &entry : block.at("cases")) {
    ReferenceCase c;
    c.components = entry.at("components").get<std::vector<std::string>>();
    c.temperature = entry.at("T").get<double>();
    auto z = entry.at("z").get<std::vector<double>>();
    auto resid = entry.at("lngamma_resid").get<std::vector<double>>();
    auto comb = entry.at("lngamma_comb").get<std::vector<double>>();
    c.mole_fractions = Eigen::Map<Vec>(z.data(), z.size());
    c.ln_gamma_residual = Eigen::Map<Vec>(resid.data(), resid.size());
    c.ln_gamma_combinatorial = Eigen::Map<Vec>(comb.data(), comb.size());
    set.cases.push_back(std::move(c));
  }
  return set;
}

void check_against_reference(const std::string &model_key,
                             const Parameters &params, double tolerance) {
  auto reference = load_reference(model_key);
  REQUIRE(!reference.cases.empty());

  double worst_residual = 0.0, worst_combinatorial = 0.0;
  for (const auto &c : reference.cases) {
    std::vector<Component> components;
    for (const auto &name : c.components)
      components.push_back(reference.components.at(name));
    REQUIRE(components.front().profile.num_classes() == params.num_classes());

    occ::solvent::sigma::PotentialOptions options;
    options.temperature = c.temperature;
    // Far tighter than the reference's 1e-8 on Gamma, so what is left is
    // its convergence error rather than ours.
    options.tolerance_mu = 1e-12;
    options.tolerance_variance = 1e-10;

    Vec residual = occ::solvent::sigma::residual_ln_gamma(
        components, c.mole_fractions, params, options);
    Vec combinatorial = occ::solvent::sigma::combinatorial_ln_gamma(
        components, c.mole_fractions, params);

    worst_residual = std::max(
        worst_residual, (residual - c.ln_gamma_residual).cwiseAbs().maxCoeff());
    worst_combinatorial =
        std::max(worst_combinatorial,
                 (combinatorial - c.ln_gamma_combinatorial).cwiseAbs().maxCoeff());
  }

  fmt::print("{}: {} cases, max |d ln gamma| residual {:.3e}, "
             "combinatorial {:.3e}\n",
             model_key, reference.cases.size(), worst_residual,
             worst_combinatorial);
  REQUIRE(worst_combinatorial < tolerance);
  REQUIRE(worst_residual < tolerance);
}

} // namespace

TEST_CASE("Damped iteration needs more than 200 steps on cold water",
          "[solvent][sigma][validation]") {
  // COSMO-SAC 2002 on nearly-pure water at low temperature is the stiffest
  // case in the reference set: the threshold H-bond term is strong and the
  // residual is a near-total cancellation between the mixture and pure
  // potentials. The reference implementation's damped loop is capped at 200
  // iterations and returns silently when it runs out, which is where the
  // 2002 tolerance below comes from.
  auto reference = load_reference("cosmo_sac_2002");
  auto params = Parameters::cosmo_sac_2002();
  const auto &water = reference.components.at("water");

  auto kernel = occ::solvent::sigma::build_kernel(water.profile.grid, params,
                                                  283.15);
  occ::solvent::sigma::PotentialOptions picard;
  picard.use_newton = false;
  picard.mixing = 0.5;
  picard.temperature = 283.15;
  picard.max_iterations = 100000;
  auto slow = occ::solvent::sigma::solve_sigma_potential(water.profile, kernel,
                                                         picard);
  REQUIRE(slow.converged);

  occ::solvent::sigma::PotentialOptions newton = picard;
  newton.use_newton = true;
  auto fast = occ::solvent::sigma::solve_sigma_potential(water.profile, kernel,
                                                          newton);
  REQUIRE(fast.converged);

  fmt::print("cold water 2002: damped {} iterations, Newton {}\n",
             slow.iterations, fast.iterations);
  REQUIRE(slow.iterations > 200);
  REQUIRE(fast.iterations < 40);
  REQUIRE((slow.mu - fast.mu).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("COSMO-SAC 2002 matches the reference implementation",
          "[solvent][sigma][validation]") {
  // Looser than 2010 because COSMO1's damped loop gives up after 200
  // iterations; cold nearly-pure water needs 601. See the test above.
  check_against_reference("cosmo_sac_2002", Parameters::cosmo_sac_2002(), 1e-4);
}

TEST_CASE("COSMO-SAC 2010 matches the reference implementation",
          "[solvent][sigma][validation]") {
  check_against_reference("cosmo_sac_2010", Parameters::cosmo_sac_2010(), 1e-6);
}

TEST_CASE("occ-generated profiles reproduce published activity coefficients",
          "[solvent][sigma][validation]") {
  // Tier 2: the machinery is fixed (validated above) and only the profiles
  // change, so the spread here is entirely our cavity and QM protocol.
  // occ profiles are B3LYP/def2-TZVP on the published geometries, Klamt
  // radii, no probe; the reference profiles are DMol3 VWN-BP/DNP.
  auto reference = load_reference("cosmo_sac_2010");
  auto params = Parameters::cosmo_sac_2010();

  ankerl::unordered_dense::map<std::string, Component> ours;
  double worst_area = 0.0, worst_volume = 0.0;
  for (const auto &[name, published] : reference.components) {
    auto file = occ::solvent::sigma::read_sigma_profile(
        std::string(OCC_TEST_DATA_DIR) + "/sigma_profiles_occ/" + name +
        ".sigma");
    REQUIRE(file.component.profile.grid == published.profile.grid);
    REQUIRE(file.component.profile.num_classes() ==
            published.profile.num_classes());
    worst_area = std::max(worst_area, std::abs(file.component.area() -
                                               published.area()) /
                                          published.area());
    worst_volume = std::max(worst_volume, std::abs(file.component.volume -
                                                   published.volume) /
                                              published.volume);
    ours.emplace(name, std::move(file.component));
  }

  double worst = 0.0, sum_squares = 0.0;
  int count = 0;
  ankerl::unordered_dense::map<std::string, double> per_system;
  for (const auto &c : reference.cases) {
    std::vector<Component> published, generated;
    for (const auto &name : c.components) {
      published.push_back(reference.components.at(name));
      generated.push_back(ours.at(name));
    }
    occ::solvent::sigma::PotentialOptions options;
    options.temperature = c.temperature;

    Vec a = occ::solvent::sigma::activity_coefficients(published,
                                                       c.mole_fractions,
                                                       params, options);
    Vec b = occ::solvent::sigma::activity_coefficients(generated,
                                                       c.mole_fractions,
                                                       params, options);
    const std::string system = c.components[0] + "/" + c.components[1];
    for (Eigen::Index i = 0; i < a.size(); i++) {
      const double d = std::abs(a(i) - b(i));
      per_system[system] = std::max(per_system[system], d);
      worst = std::max(worst, d);
      sum_squares += d * d;
      count++;
    }
  }

  for (const auto &[system, value] : per_system)
    fmt::print("   {:24s} max |d ln gamma| {:.3f}\n", system, value);
  fmt::print("protocol spread over {} values: max |d ln gamma| {:.3f}, "
             "rms {:.3f}; cavity area {:.1f}%, volume {:.1f}%\n",
             count, worst, std::sqrt(sum_squares / count), 100 * worst_area,
             100 * worst_volume);

  // Our cavity tracks the published one to a few percent.
  REQUIRE(worst_area < 0.06);
  REQUIRE(worst_volume < 0.05);

  // A B3LYP/def2-TZVP conductor cavity stands in for DMol3 VWN-BP/DNP to
  // within a few tenths of a log unit everywhere, and better than that away
  // from water. Pinned so a protocol regression shows up.
  REQUIRE(per_system.at("water/acetone") < 0.9);
  for (const auto &system : {"water/ethanol", "methanol/benzene",
                             "benzene/n-hexane", "acetone/chloroform"})
    REQUIRE(per_system.at(system) < 0.25);
  REQUIRE(worst < 0.9);
  REQUIRE(std::sqrt(sum_squares / count) < 0.2);
}

namespace {

// Parse the DMol3 .cosmo output: the BIOSYM car block for the atoms, and the
// segment table (position in Bohr, charge in e, area in Angstrom^2).
struct Dmol3Cosmo {
  occ::IVec atomic_numbers;
  Mat3N atom_positions_bohr;
  Segments segments;
};

Dmol3Cosmo read_dmol3_cosmo(const std::string &path) {
  std::ifstream input(path);
  REQUIRE(input.good());
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);)
    lines.push_back(line);

  std::vector<int> numbers;
  std::vector<Vec3> atoms;
  std::vector<Vec3> seg_positions;
  std::vector<double> seg_charge, seg_area;
  std::vector<int> seg_atom;

  bool in_car = false, in_segments = false;
  for (size_t i = 0; i < lines.size(); i++) {
    const auto &line = lines[i];
    if (line.find("!BIOSYM archive") != std::string::npos) { in_car = true; continue; }
    if (in_car && line.rfind("end", 0) == 0) { in_car = false; continue; }
    if (line.find("charge/area") != std::string::npos) { in_segments = true; continue; }

    std::istringstream row(line);
    if (in_car) {
      std::string label;
      double x, y, z;
      if (row >> label >> x >> y >> z) {
        std::string element;
        for (char ch : label) {
          if (std::isalpha(static_cast<unsigned char>(ch)))
            element.push_back(ch);
          else
            break;
        }
        numbers.push_back(occ::core::Element(element).atomic_number());
        atoms.emplace_back(x, y, z);
      }
    } else if (in_segments) {
      int n = 0, atom = 0;
      double x, y, z, charge, area, charge_area, potential;
      if (row >> n >> atom >> x >> y >> z >> charge >> area >> charge_area >>
          potential) {
        seg_positions.emplace_back(x, y, z);
        seg_charge.push_back(charge);
        seg_area.push_back(area);
        seg_atom.push_back(atom - 1);
      }
    }
  }

  REQUIRE(!atoms.empty());
  REQUIRE(!seg_area.empty());

  Dmol3Cosmo out;
  const int natoms = static_cast<int>(atoms.size());
  out.atomic_numbers = occ::IVec(natoms);
  out.atom_positions_bohr = Mat3N(3, natoms);
  for (int i = 0; i < natoms; i++) {
    out.atomic_numbers(i) = numbers[i];
    out.atom_positions_bohr.col(i) = atoms[i] * occ::units::ANGSTROM_TO_BOHR;
  }

  const int nseg = static_cast<int>(seg_area.size());
  out.segments.positions = Mat3N(3, nseg);
  out.segments.areas = Vec(nseg);
  out.segments.sigma = Vec(nseg);
  out.segments.atom_index = occ::IVec(nseg);
  out.segments.atomic_number = occ::IVec(nseg);
  for (int i = 0; i < nseg; i++) {
    out.segments.positions.col(i) = seg_positions[i]; // already Bohr
    out.segments.areas(i) = seg_area[i];
    out.segments.sigma(i) = seg_charge[i] / seg_area[i];
    out.segments.atom_index(i) = seg_atom[i];
    out.segments.atomic_number(i) = numbers[seg_atom[i]];
  }
  return out;
}

} // namespace

TEST_CASE("Published segments reproduce the published profile",
          "[solvent][sigma][validation]") {
  // The strongest available check on the sigma machinery: take DMol3's own
  // segment charges and push them through occ's averaging, H-bond
  // classification and binning. Agreement here means averaging, the
  // classification rules, the fractional split and the deposit are all
  // exactly right, and any remaining difference against the published
  // profiles comes from the screening charges themselves.
  auto reference = load_reference("cosmo_sac_2010");
  auto params = Parameters::cosmo_sac_2010();
  Grid grid;

  for (const auto &name : {"water", "ethanol"}) {
    auto parsed = read_dmol3_cosmo(std::string(OCC_TEST_DATA_DIR) + "/" +
                                   name + "_dmol3.cosmo");
    occ::solvent::sigma::classify_hbond_segments(
        parsed.segments, parsed.atomic_numbers, parsed.atom_positions_bohr);
    occ::solvent::sigma::average_sigma(parsed.segments, params.r_av,
                                       params.f_decay);
    auto rebuilt = occ::solvent::sigma::bin_segments(parsed.segments, grid,
                                                     params.hbond_split());

    const auto &published = reference.components.at(name).profile;
    const double area = published.total_area();
    const double l1 =
        (rebuilt.values - published.values).cwiseAbs().sum() / area;
    fmt::print("{}: rebuilt from published segments, area {:.4f} vs {:.4f}, "
               "normalised L1 {:.3e}\n",
               name, rebuilt.total_area(), area, l1);

    REQUIRE(rebuilt.total_area() == Catch::Approx(area).epsilon(1e-3));
    REQUIRE(l1 < 1e-10);
  }
}

TEST_CASE("Substituting one component at a time is well behaved",
          "[solvent][sigma][validation]") {
  auto reference = load_reference("cosmo_sac_2010");
  auto params = Parameters::cosmo_sac_2010();

  ankerl::unordered_dense::map<std::string, Component> ours;
  for (const auto &name : {"water", "ethanol"})
    ours.emplace(name, occ::solvent::sigma::read_sigma_profile(
                           std::string(OCC_TEST_DATA_DIR) +
                           "/sigma_profiles_occ/" + name + ".sigma")
                           .component);

  occ::solvent::sigma::PotentialOptions options;
  options.temperature = 283.15;
  Vec x(2);
  x << 0.05, 0.95;

  fmt::print("\n water/ethanol, x_water=0.05, T=283.15 -- ln gamma(water)\n");
  for (bool occ_water : {false, true}) {
    for (bool occ_ethanol : {false, true}) {
      std::vector<Component> c{
          occ_water ? ours.at("water") : reference.components.at("water"),
          occ_ethanol ? ours.at("ethanol")
                      : reference.components.at("ethanol")};
      Vec g = occ::solvent::sigma::activity_coefficients(c, x, params, options);
      Vec r = occ::solvent::sigma::residual_ln_gamma(c, x, params, options);
      fmt::print("   water={:3s} ethanol={:3s} : total {:+9.4f}  residual "
                 "{:+9.4f}\n",
                 occ_water ? "occ" : "UD", occ_ethanol ? "occ" : "UD", g(0),
                 r(0));
      // No combination blows up. An unguarded Newton step used to send the
      // occ-ethanol cases to +9.6 instead of staying in this range.
      //
      // Note the mixed rows sit further from the all-UD baseline than the
      // all-occ row does: protocol errors partly cancel when both profiles
      // come from the same source, which is a reason never to mix
      // occ-generated and published profiles in one mixture.
      REQUIRE(g(0) == Catch::Approx(0.4365).margin(0.35));
    }
  }
}

TEST_CASE("Moment profiles converge under grid refinement",
          "[solvent][sigma][validation]") {
  // The 2010 exchange kernel is discontinuous at sigma = 0 (the H-bond sign
  // gate), and that sits in the middle of the grid where p(sigma) is largest.
  // The second moment is the quantity that cares, so refine and watch it.
  auto parsed = read_dmol3_cosmo(std::string(OCC_TEST_DATA_DIR) +
                                 "/water_dmol3.cosmo");
  auto params = Parameters::cosmo_sac_2010();
  occ::solvent::sigma::classify_hbond_segments(
      parsed.segments, parsed.atomic_numbers, parsed.atom_positions_bohr);
  occ::solvent::sigma::average_sigma(parsed.segments, params.r_av,
                                     params.f_decay);

  // Probe points away from and astride the discontinuity.
  const std::vector<double> probes{-0.018, -0.012, -0.004, 0.0,
                                   0.004,  0.012,  0.018};

  auto evaluate = [&](int n) {
    Grid grid{n, -0.025, 0.025};
    auto profile = occ::solvent::sigma::bin_segments(parsed.segments, grid,
                                                     params.hbond_split());
    auto kernel = occ::solvent::sigma::build_kernel(grid, params, 298.15);
    auto potential =
        occ::solvent::sigma::solve_sigma_potential(profile, kernel);
    REQUIRE(potential.converged);

    // Sample mu and variance of the OH column at the probe points.
    const int oh = static_cast<int>(HBondClass::OH);
    Mat mu_field = potential.mu;
    Mat var_field = potential.variance;
    Segments probe_segments;
    probe_segments.positions = Mat3N::Zero(3, probes.size());
    probe_segments.areas = Vec::Ones(probes.size());
    probe_segments.sigma = Eigen::Map<const Vec>(probes.data(), probes.size());
    probe_segments.sigma_averaged = probe_segments.sigma;
    probe_segments.atom_index = occ::IVec::Zero(probes.size());
    probe_segments.atomic_number = occ::IVec::Constant(probes.size(), 8);
    probe_segments.hbond_class = occ::IVec::Constant(probes.size(), oh);

    Vec mu = occ::solvent::sigma::contract_segments(probe_segments, grid,
                                                    mu_field);
    Vec var = occ::solvent::sigma::contract_segments(probe_segments, grid,
                                                     var_field);
    return std::make_pair(mu, var);
  };

  auto finest = evaluate(801);
  fmt::print("\n grid refinement against n=801 (water, COSMO-SAC 2010)\n");
  fmt::print("{:>6} {:>7} {:>14} {:>14}\n", "n", "0 node", "max |d mu|",
             "max |d Var|");
  double mu_error = 0.0, var_error = 0.0;
  // Odd n puts sigma = 0 exactly on a node; even n straddles it, which is
  // the harsher case for the H-bond sign gate.
  for (int n : {50, 51, 100, 101, 200, 201, 400, 401}) {
    auto sampled = evaluate(n);
    const double d_mu = (sampled.first - finest.first).cwiseAbs().maxCoeff();
    const double d_var = (sampled.second - finest.second).cwiseAbs().maxCoeff();
    fmt::print("{:>6} {:>7} {:14.5f} {:14.5f}\n", n, (n % 2) ? "yes" : "no",
               d_mu, d_var);
    if (n >= 400) {
      mu_error = std::max(mu_error, d_mu);
      var_error = std::max(var_error, d_var);
    }
  }

  // At the finest step below, both moments must have settled.
  REQUIRE(mu_error < 0.02);
  REQUIRE(var_error < 0.05);
}

TEST_CASE("Conductor cavity cost scales usably", "[.][solvent][sigma][scaling]") {
  // The conductor cavity builds a dense ncav x ncav COSMO matrix and factors
  // it. cg targets drug-sized molecules, so check where that lands before
  // building on top of it. Hidden by default; run with [scaling].
  struct Case { const char *name; const char *xyz; };
  const std::vector<Case> cases{{"water", WATER},
                                {"naphthol", NAPHTHOL},
                                {"desloratadine", DESLORATADINE}};

  fmt::print("\n{:>15} {:>6} {:>8} {:>9} {:>10} {:>10}\n", "molecule", "atoms",
             "n_ang", "segments", "build (s)", "solve (s)");
  for (const auto &c : cases) {
    auto mol = occ::io::molecule_from_xyz_string(c.xyz);
    Mat3N pos = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
    occ::IVec nums = mol.atomic_numbers();
    for (int n_ang : {146, 590}) {
      occ::scrf::ReactionFieldEngine engine(
          occ::scrf::Options::conductor(0.0, n_ang));
      auto t0 = std::chrono::steady_clock::now();
      engine.initialize(pos, nums);
      auto t1 = std::chrono::steady_clock::now();
      Vec phi = Vec::Random(engine.num_es_surface_points()) * 0.01;
      engine.solve_asc(phi);
      auto t2 = std::chrono::steady_clock::now();
      const double build =
          std::chrono::duration<double>(t1 - t0).count();
      const double solve =
          std::chrono::duration<double>(t2 - t1).count();
      fmt::print("{:>15} {:>6} {:>8} {:>9} {:>10.2f} {:>10.3f}\n", c.name,
                 mol.size(), n_ang, engine.num_es_surface_points(), build,
                 solve);
      REQUIRE(engine.num_es_surface_points() > 0);
    }
  }
}

TEST_CASE("Charged solutes screen to minus the solute charge",
          "[solvent][sigma][conductor]") {
  // In a conductor the surface charge must integrate to -q_solute exactly;
  // the shortfall is the outlying-charge error, which is far more visible on
  // an ion than on a neutral molecule.
  struct Ion { const char *name; const char *xyz; int charge; };
  for (const auto &ion : {Ion{"ammonium", AMMONIUM, 1},
                          Ion{"formate", FORMATE, -1}}) {
    auto mol = occ::io::molecule_from_xyz_string(ion.xyz);
    occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
    basis.set_pure(true);
    occ::dft::DFT gas_ks("b3lyp", basis);
    occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
    gas_scf.set_charge_multiplicity(ion.charge, 1);
    gas_scf.compute_scf_energy();

    occ::driver::SigmaProfileSettings settings;
    settings.basis = "def2-svp";
    auto result = occ::driver::conductor_profile(gas_scf.wavefunction(), settings);

    const double expected = -static_cast<double>(ion.charge);
    const double shortfall = result.screening_charge - expected;
    fmt::print("{}: q={:+d}, screening charge {:+.5f} (exact {:+.1f}), "
               "shortfall {:+.5f} e = {:.2f}%\n",
               ion.name, ion.charge, result.screening_charge, expected,
               shortfall, 100.0 * std::abs(shortfall));

    REQUIRE(result.segments.size() > 100);
    // Sign and magnitude must be right; the residual is outlying charge.
    REQUIRE(std::abs(shortfall) < 0.15);
    REQUIRE(result.energy_conductor < result.energy_gas);
  }
}

TEST_CASE("Charge constraint restores the sum rule without reshaping sigma",
          "[solvent][sigma][conductor]") {
  // Gauss's law fixes the total; the question is whether enforcing it also
  // moves the distribution, which is what the profile actually is.
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
  basis.set_pure(true);
  occ::dft::DFT gas_ks("b3lyp", basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
  gas_scf.compute_scf_energy();
  auto wfn = gas_scf.wavefunction();

  auto params = Parameters::cosmo_sac_2010();
  auto free_segments =
      occ::driver::conductor_segments(wfn, params, 0.0, 590, false);
  auto fixed_segments =
      occ::driver::conductor_segments(wfn, params, 0.0, 590, true);

  REQUIRE(free_segments.size() == fixed_segments.size());
  fmt::print("\n water: screening charge free {:+.5f}, constrained {:+.5f}\n",
             free_segments.total_charge(), fixed_segments.total_charge());
  REQUIRE(std::abs(fixed_segments.total_charge()) < 1e-10);

  const double max_shift =
      (free_segments.sigma - fixed_segments.sigma).cwiseAbs().maxCoeff();
  const double sigma_scale = free_segments.sigma.cwiseAbs().maxCoeff();
  fmt::print(" max |d sigma| {:.3e} e/A^2 ({:.2f}% of peak sigma)\n", max_shift,
             100 * max_shift / sigma_scale);

  Grid grid;
  auto free_profile = occ::solvent::sigma::bin_segments(free_segments, grid,
                                                        params.hbond_split());
  auto fixed_profile = occ::solvent::sigma::bin_segments(fixed_segments, grid,
                                                         params.hbond_split());
  const double l1 = (free_profile.values - fixed_profile.values)
                        .cwiseAbs()
                        .sum() /
                    free_profile.total_area();
  fmt::print(" normalised L1 between the two profiles {:.4f}\n", l1);

  // The multiplier adds a smooth, near-uniform field, so the shape barely
  // moves. Anything large here would mean the constraint is doing more than
  // restoring the sum rule.
  REQUIRE(l1 < 0.05);
}

TEST_CASE("draco", "[solvent]") {
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  auto nums = mol.atomic_numbers();
  auto pos = mol.positions() * occ::units::ANGSTROM_TO_BOHR;
  auto params = occ::solvent::get_smd_parameters("toluene");

  Vec cn = occ::solvent::draco::coordination_numbers(nums, pos);
  fmt::print("Coordination numbers:\n{}\n", format_matrix(cn));
  Vec q = occ::core::charges::eeq_partial_charges(nums, pos, 0.0);

  Vec radii = occ::solvent::draco::smd_coulomb_radii(q, nums, pos, params);
}

TEST_CASE("Incremental Fock: solvation opts out on the core Hamiltonian",
          "[scf][incremental]") {
  // Solvation leaves the two-electron build perfectly linear — it is H that
  // moves, because `update_core_hamiltonian` folds in the apparent surface
  // charge every cycle. Forwarding a single "supports incremental" bool from
  // the wrapped method missed this entirely.
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::gto::AOBasis::load(atoms, "6-31G");
  occ::qm::HartreeFock hf(basis);
  REQUIRE(occ::qm::supports_incremental_fock_build(hf.fock_build_properties()));

  occ::solvent::SolvationCorrectedProcedure<occ::qm::HartreeFock> solv(hf,
                                                                       "water");
  auto properties = solv.fock_build_properties();
  REQUIRE(properties.linear_in_density);
  REQUIRE_FALSE(properties.constant_core_hamiltonian);
  REQUIRE_FALSE(occ::qm::supports_incremental_fock_build(properties));

  occ::qm::SCF<occ::solvent::SolvationCorrectedProcedure<occ::qm::HartreeFock>>
      scf(solv);
  REQUIRE_FALSE(scf.incremental_fock_supported());
  // ...and no caller can undo that by reaching into the settings.
  scf.convergence_settings.incremental_fock_threshold = 1.0;
  REQUIRE_FALSE(scf.incremental_fock_supported());
}
