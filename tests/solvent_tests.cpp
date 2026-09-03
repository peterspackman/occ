#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fstream>
#include <map>
#include <occ/cg/distance_partition.h>
#include <occ/core/eeq.h>
#include <occ/core/element.h>
#include <occ/core/format_matrix.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/driver/cosmors_driver.h>
#include <occ/driver/cosmors_solvation.h>
#include <occ/io/xyz.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/cosmors.h>
#include <occ/solvent/cosmors_io.h>
#include <occ/solvent/cosmors_segments.h>
#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/solvent/surface.h>

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

  double va = occ::solvent::surface::cavity_volume(make_cavity(radii, a, 0.0),
                                                   cavity_atom_positions(a));
  double vb = occ::solvent::surface::cavity_volume(make_cavity(radii, b, 0.0),
                                                   cavity_atom_positions(b));
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

// Segment descriptors

namespace {

using occ::solvent::cosmors::Segments;

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
  for (int i = 0; i < n; i++) {
    s.positions(0, i) = x_angs[i] * occ::units::ANGSTROM_TO_BOHR;
    s.sigma(i) = sigmas[i];
  }
  return s;
}

} // namespace

TEST_CASE("Segments carry charge density, not charge", "[solvent][cosmors]") {
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
  auto s = occ::solvent::cosmors::segments_from_cavity(cavity, charges, nums);
  const double conv =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  REQUIRE(s.areas(0) == Catch::Approx(conv));
  REQUIRE(s.sigma(0) == Catch::Approx(-0.3 / conv));
  REQUIRE(s.sigma(1) == Catch::Approx(0.1 / (4.0 * conv)));
  // Total screening charge is recovered exactly.
  REQUIRE(s.total_charge() == Catch::Approx(-0.2));
}

TEST_CASE("Segment averaging is a local weighted mean", "[solvent][cosmors]") {
  auto s = line_segments({0.0, 0.05, 10.0}, {0.01, -0.01, 0.05}, 0.35);
  occ::solvent::cosmors::average_sigma(s, 0.5);

  // Two near-coincident segments average to each other.
  REQUIRE(s.sigma_averaged(0) ==
          Catch::Approx(s.sigma_averaged(1)).margin(1e-3));
  REQUIRE(std::abs(s.sigma_averaged(0)) < 2e-3);
  // An isolated segment keeps its own value.
  REQUIRE(s.sigma_averaged(2) == Catch::Approx(0.05).epsilon(1e-6));
  // A weighted mean never leaves the range of its inputs.
  REQUIRE(s.sigma_averaged.minCoeff() >= s.sigma.minCoeff() - 1e-12);
  REQUIRE(s.sigma_averaged.maxCoeff() <= s.sigma.maxCoeff() + 1e-12);
}

TEST_CASE("Averaging a constant field is the identity", "[solvent][cosmors]") {
  auto s =
      line_segments({0.0, 0.3, 0.6, 0.9}, {0.007, 0.007, 0.007, 0.007}, 0.35);
  occ::solvent::cosmors::average_sigma(s, 0.5);
  for (Eigen::Index i = 0; i < s.size(); i++)
    REQUIRE(s.sigma_averaged(i) == Catch::Approx(0.007).epsilon(1e-12));
}

TEST_CASE("Ideal-conductor COSMO profile for water",
          "[solvent][cosmors][conductor]") {
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  auto atoms = mol.atoms();

  occ::gto::AOBasis basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(true);
  occ::dft::DFT gas_ks("b3lyp", basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
  double gas_energy = gas_scf.compute_scf_energy();
  auto gas_wfn = gas_scf.wavefunction();

  occ::driver::ConductorSettings settings;
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
}

#ifndef OCC_DATA_DIR
#define OCC_DATA_DIR "share"
#endif

namespace {

/// The segment ensembles occ ships, which occ generates itself
/// (`occ cosmo-rs --write-segments`) at the driver's default level of
/// theory. Resolved from the source tree rather than `OCC_DATA_PATH` so the
/// ensembles under test are the ones in the checkout.
occ::solvent::cosmors::SegmentStore shipped_segments() {
  return occ::solvent::cosmors::SegmentStore(
      {std::string(OCC_DATA_DIR) + "/solvent/cosmors"});
}

} // namespace

TEST_CASE("openCOSMO-RS 24a defaults are the published parameter set",
          "[solvent][cosmors][parameters]") {
  // The kernel, the combinatorial term and the free-energy assembly were
  // regressed together, so `Parameters` carries all three and its defaults
  // are the 24a set published in Grigorash et al., J. Comput. Chem. 2024,
  // 45, 2699. The struct's member initialisers are the only copy of the
  // values in the source; this pins them so an accidental edit fails here
  // rather than silently becoming a different model.
  const auto p = occ::solvent::cosmors::Parameters::v24a();

  REQUIRE(p.a_eff == Catch::Approx(5.9248470));
  REQUIRE(p.r_av == Catch::Approx(0.5));
  REQUIRE(p.r_corr == Catch::Approx(1.0));
  REQUIRE(p.sigma_orth_factor == Catch::Approx(0.816));

  REQUIRE(p.mf_alpha == Catch::Approx(7.2847361e06));
  REQUIRE(p.mf_f_corr == Catch::Approx(2.4));
  REQUIRE(p.hb_c == Catch::Approx(4.3311555e07));
  REQUIRE(p.hb_c_T == Catch::Approx(1.5));
  REQUIRE(p.hb_t_ref == Catch::Approx(298.15));
  REQUIRE(p.hb_sigma_thresh == Catch::Approx(9.6112460e-03));

  REQUIRE(p.comb_z == Catch::Approx(10.0));
  REQUIRE(p.comb_a_std == Catch::Approx(4.1623570e01));

  REQUIRE(p.eta == Catch::Approx(-4.448499));
  REQUIRE(p.omega_ring == Catch::Approx(2.6302510e-01));

  // Per-element van der Waals surface tensions, kcal/mol/A^2.
  const std::vector<std::pair<int, double>> tau{
      {1, 2.933803e-02},  {6, 2.287904e-02},  {7, 7.007681e-04},
      {8, 3.545052e-03},  {9, 5.608829e-03},  {14, 4.215503e-03},
      {15, 3.607977e-03}, {16, 3.498700e-02}, {17, 3.414282e-02},
      {35, 4.085111e-02},
  };
  REQUIRE(p.tau.size() == tau.size());
  for (const auto &[z, value] : tau) {
    INFO("element " << z);
    REQUIRE(p.tau.contains(z));
    REQUIRE(p.tau.at(z) == Catch::Approx(value));
  }
}

TEST_CASE("openCOSMO-RS segment ensembles survive a round trip",
          "[solvent][cosmors]") {
  const auto params = occ::solvent::cosmors::Parameters::v24a();
  const auto store = shipped_segments();
  REQUIRE(store.contains("acetone"));
  const auto original = store.get("acetone").component;

  const auto path =
      (std::filesystem::temp_directory_path() / "occ_rsseg_roundtrip.rsseg")
          .string();
  occ::solvent::cosmors::write_segments(path, "acetone", original, params,
                                        "b3lyp", "def2-svp");
  auto loaded = occ::solvent::cosmors::read_segments(path);
  std::filesystem::remove(path);

  REQUIRE(loaded.name == "acetone");
  REQUIRE(loaded.method == "b3lyp");
  REQUIRE(loaded.basis == "def2-svp");
  REQUIRE(loaded.r_av == params.r_av);
  REQUIRE(loaded.r_corr == params.r_corr);
  REQUIRE(loaded.component.size() == original.size());
  // Written at 14 significant figures, so the descriptors come back to
  // relative 1e-13 and the energies that follow are unaffected.
  REQUIRE((loaded.component.sigma - original.sigma).cwiseAbs().maxCoeff() <
          1e-15);
  REQUIRE((loaded.component.sigma_orth - original.sigma_orth)
              .cwiseAbs()
              .maxCoeff() < 1e-15);
  REQUIRE((loaded.component.area - original.area).cwiseAbs().maxCoeff() <
          1e-15);
  REQUIRE((loaded.component.atomic_number - original.atomic_number)
              .cwiseAbs()
              .sum() == 0);
  REQUIRE(std::abs(loaded.component.total_area() - original.total_area()) <
          1e-12);
  REQUIRE(std::abs(loaded.component.volume - original.volume) < 1e-12);

  // The point of the file is that a solvent loaded from it behaves like one
  // built in memory.
  occ::solvent::cosmors::ActivityOptions options;
  occ::solvent::cosmors::SolventModel from_memory(original, params, options);
  occ::solvent::cosmors::SolventModel from_file(loaded.component, params,
                                                options);
  REQUIRE(std::abs(from_memory.residual_energy(original) -
                   from_file.residual_energy(loaded.component)) < 1e-14);
}

TEST_CASE("openCOSMO-RS mixtures reduce to the pure component",
          "[solvent][cosmors]") {
  // Mixing a component with itself must be indistinguishable from the pure
  // component at any composition, which pins the area weighting and the
  // volume/cavity-area averaging in mix_components.
  const auto params = occ::solvent::cosmors::Parameters::v24a();
  const auto store = shipped_segments();
  const auto water = store.get("water").component;
  const auto solute = store.get("methanol").component;

  occ::solvent::cosmors::ActivityOptions options;
  occ::solvent::cosmors::SolventModel pure(water, params, options);
  const double reference = pure.residual_energy(solute);

  for (double x : {0.25, 0.5, 0.75}) {
    INFO(x);
    Vec fractions(2);
    fractions << x, 1.0 - x;
    auto mixed =
        occ::solvent::cosmors::mix_components({water, water}, fractions);
    REQUIRE(std::abs(mixed.volume - water.volume) < 1e-10);
    REQUIRE(std::abs(mixed.total_area() - water.total_area()) < 1e-10);
    occ::solvent::cosmors::SolventModel model(mixed, params, options);
    REQUIRE(std::abs(model.residual_energy(solute) - reference) < 1e-12);
  }
}

TEST_CASE("openCOSMO-RS infinite dilution matches the mixture solve",
          "[solvent][cosmors]") {
  // The infinite-dilution test particle and the mixture solve are separate
  // code paths that must agree where they overlap: a vanishing solute mole
  // fraction.
  const auto params = occ::solvent::cosmors::Parameters::v24a();
  occ::solvent::cosmors::ActivityOptions options;
  options.temperature = 298.15;

  const auto store = shipped_segments();
  const auto water = store.get("water").component;
  occ::solvent::cosmors::SolventModel solvent(water, params, options);

  for (const char *name : {"methanol", "acetone"}) {
    INFO(name);
    const auto solute = store.get(name).component;
    const std::vector<occ::solvent::cosmors::Component> pair{solute, water};
    Vec x(2);
    x << 1e-10, 1.0 - 1e-10;
    const double mixture =
        occ::solvent::cosmors::residual_ln_gamma(pair, x, params, options)(0);
    const double rt = occ::constants::molar_gas_constant<double> *
                      options.temperature / 1000.0;
    const double test_particle = solvent.residual_energy(solute) *
                                 occ::units::AU_TO_KJ_PER_MOL / rt;
    REQUIRE(std::abs(mixture - test_particle) < 1e-8);
  }
}

TEST_CASE("COSMO-RS solvation free energy runs end to end",
          "[solvent][cosmors][driver]") {
  // Covers the one-call facade the CLI and every language binding sit on:
  // gas SCF, conductor SCF, segment descriptors, cached solvent ensemble,
  // and the free-energy assembly. Water is the cheapest solute that
  // exercises all of it, and the shipped ensembles are built at the
  // settings' defaults so solute and solvent are at the same level.
  //
  // Experimental geometry, r(OH) = 0.9572 A and 104.52 degrees.
  const auto solute = occ::io::molecule_from_xyz_string(R"(3
water
O   0.000000   0.000000   0.000000
H   0.756950  -0.585880   0.000000
H  -0.756950  -0.585880   0.000000
)");

  occ::driver::CosmoRSSolvationSettings settings;
  settings.liquid_volume = 30.01; // A^3 per molecule at 298 K
  settings.num_rings = 0;
  const auto result =
      occ::driver::cosmors_solvation_free_energy(solute, "water", settings);

  const double k = occ::units::AU_TO_KJ_PER_MOL;
  fmt::print("\nopenCOSMO-RS 24a: water in water (kJ/mol)\n");
  fmt::print("  dielectric {:8.2f}\n  residual   {:8.2f}\n"
             "  combinat.  {:8.2f}\n  vdw        {:8.2f}\n"
             "  ref. state {:8.2f}\n  eta        {:8.2f}\n  total      "
             "{:8.2f}  (experiment -26.4)\n\n",
             result.energy.dielectric * k, result.energy.residual * k,
             result.energy.combinatorial * k, result.energy.vdw * k,
             result.energy.reference_state * k, result.energy.eta * k,
             result.total() * k);

  REQUIRE(result.cavity_area > 0.0);
  REQUIRE(result.cavity_volume > 0.0);
  REQUIRE(result.num_rings == 0);
  // Screening a neutral dipole against an ideal conductor stabilises it.
  REQUIRE(result.energy.dielectric < 0.0);
  // The van der Waals term is -sum(tau * area) with positive tau.
  REQUIRE(result.energy.vdw < 0.0);

  // Loose: this checks the assembly is on the right scale and has the right
  // sign against the Minnesota solvation database value, not that the model
  // is accurate to the last kJ/mol.
  REQUIRE(std::abs(result.total() * k - (-26.4)) < 25.0);
}

TEST_CASE("Cached solvent ensembles are discoverable", "[solvent][cosmors]") {
  // available_cosmors_solvents is what the CLI and the bindings list, so it
  // has to see the ensembles that ship with occ.
  const auto store = shipped_segments();
  const auto names = store.available();
  REQUIRE(names.size() >= 5);
  REQUIRE(std::is_sorted(names.begin(), names.end()));
  for (const char *expected : {"water", "methanol", "acetone", "benzene"}) {
    INFO(expected);
    REQUIRE(std::find(names.begin(), names.end(), expected) != names.end());
  }
  REQUIRE_FALSE(store.contains("not-a-solvent"));
}

TEST_CASE("Conductor cavity cost scales usably",
          "[.][solvent][cosmors][scaling]") {
  // The conductor cavity builds a dense ncav x ncav COSMO matrix and factors
  // it. cg targets drug-sized molecules, so check where that lands before
  // building on top of it. Hidden by default; run with [scaling].
  struct Case {
    const char *name;
    const char *xyz;
  };
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
      const double build = std::chrono::duration<double>(t1 - t0).count();
      const double solve = std::chrono::duration<double>(t2 - t1).count();
      fmt::print("{:>15} {:>6} {:>8} {:>9} {:>10.2f} {:>10.3f}\n", c.name,
                 mol.size(), n_ang, engine.num_es_surface_points(), build,
                 solve);
      REQUIRE(engine.num_es_surface_points() > 0);
    }
  }
}

TEST_CASE("Charged solutes screen to minus the solute charge",
          "[solvent][cosmors][conductor]") {
  // In a conductor the surface charge must integrate to -q_solute exactly;
  // the shortfall is the outlying-charge error, which is far more visible on
  // an ion than on a neutral molecule.
  struct Ion {
    const char *name;
    const char *xyz;
    int charge;
  };
  for (const auto &ion :
       {Ion{"ammonium", AMMONIUM, 1}, Ion{"formate", FORMATE, -1}}) {
    auto mol = occ::io::molecule_from_xyz_string(ion.xyz);
    occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
    basis.set_pure(true);
    occ::dft::DFT gas_ks("b3lyp", basis);
    occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
    gas_scf.set_charge_multiplicity(ion.charge, 1);
    gas_scf.compute_scf_energy();

    occ::driver::ConductorSettings settings;
    settings.basis = "def2-svp";
    auto result =
        occ::driver::conductor_profile(gas_scf.wavefunction(), settings);

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
          "[solvent][cosmors][conductor]") {
  // Gauss's law fixes the total; the question is whether enforcing it also
  // moves the distribution, which is what the profile actually is.
  auto mol = occ::io::molecule_from_xyz_string(WATER);
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
  basis.set_pure(true);
  occ::dft::DFT gas_ks("b3lyp", basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas_ks);
  gas_scf.compute_scf_energy();
  auto wfn = gas_scf.wavefunction();

  const occ::solvent::cosmors::Parameters params;
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

  // The multiplier adds a smooth, near-uniform field, so the distribution
  // barely moves. Anything large here would mean the constraint is doing more
  // than restoring the sum rule.
  const double l1 =
      (free_segments.areas.array() *
       (free_segments.sigma - fixed_segments.sigma).array().abs())
          .sum() /
      (free_segments.areas.array() * free_segments.sigma.array().abs()).sum();
  fmt::print(" normalised L1 between the two distributions {:.4f}\n", l1);
  REQUIRE(l1 < 0.05);
}

namespace {

occ::crystal::Crystal acetic_acid_crystal() {
  const std::vector<std::string> labels = {"C1", "C2", "H1", "H2",
                                           "H3", "H4", "O1", "O2"};
  occ::IVec nums(labels.size());
  Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++)
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200,
      0.05100, -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900,
      0.05300, 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030,
      0.17690;
  occ::crystal::AsymmetricUnit asym(positions.transpose(), nums, labels);
  occ::crystal::SpaceGroup sg(33);
  auto cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
  return occ::crystal::Crystal(asym, sg, cell);
}

} // namespace

TEST_CASE("openCOSMO-RS cg surfaces partition over crystal neighbours",
          "[solvent][cosmors][solvation]") {
  // The end-to-end cg path: conductor SCF -> segment descriptors -> the
  // per-segment channels -> the nearest-neighbour partitioner. What matters
  // is that the surface-additive channels survive the partition intact,
  // which is exactly what segment additivity buys, and that the molecular
  // total is the model's whole free energy rather than the channels alone.
  auto crystal = acetic_acid_crystal();
  auto molecules = crystal.symmetry_unique_molecules();
  auto dimers = crystal.symmetry_unique_dimers(7.0);
  const auto &neighbors = dimers.molecule_neighbors[0];

  std::vector<occ::qm::Wavefunction> gas;
  for (const auto &mol : molecules) {
    occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "def2-svp");
    basis.set_pure(true);
    occ::dft::DFT ks("b3lyp", basis);
    occ::qm::SCF<occ::dft::DFT> scf(ks);
    scf.compute_scf_energy();
    gas.push_back(scf.wavefunction());
  }

  const auto basename =
      (std::filesystem::temp_directory_path() / "occ_cosmors_cg").string();
  occ::driver::CosmoRSSettings settings;
  settings.method = "b3lyp";
  settings.basis = "def2-svp";
  settings.pure_spherical = true;
  auto spec = occ::driver::SolventSpec::parse("water");

  auto result =
      occ::driver::cosmors_solvation(basename, molecules, gas, spec, settings);
  for (size_t i = 0; i < molecules.size(); i++)
    std::filesystem::remove(
        fmt::format("{}_{}_conductor.owf.json", basename, i));

  REQUIRE(result.surfaces.size() == molecules.size());
  const auto &data = result.surfaces[0];
  REQUIRE(data.cavities.size() == 1);
  const auto &cavity = data.cavities.front();
  REQUIRE(cavity.name == "conductor");

  // The three surface-additive terms plus the area-spread relaxation.
  std::vector<std::string> channels;
  for (const auto &e : cavity.energies)
    channels.push_back(e.name);
  std::sort(channels.begin(), channels.end());
  REQUIRE(channels == std::vector<std::string>{"dielectric", "electronic",
                                               "residual", "vdw"});

  occ::cg::SolventSurfacePartitioner partitioner(neighbors);
  partitioner.set_should_write_surface_files(false);
  partitioner.set_use_normalized_distance(false);
  auto contributions = partitioner.partition(neighbors, data);
  REQUIRE(contributions.size() == neighbors.size());

  // Every channel is conserved: the partition redistributes, it does not
  // create or destroy.
  for (const auto &channel : cavity.energies) {
    double partitioned = 0.0;
    for (const auto &c : contributions)
      partitioned += c.energy(channel.name).forward;
    fmt::print(" {:<12s} total {:12.6f}, partitioned {:12.6f} Hartree\n",
               channel.name, channel.values.sum(), partitioned);
    REQUIRE(partitioned == Catch::Approx(channel.values.sum()).margin(1e-10));
  }

  // The reported total carries the per-molecule terms too, so it is strictly
  // more negative than the channels alone by the combinatorial, ring,
  // reference-state and eta contributions.
  const double channels_only = data.total_energy();
  fmt::print(" channels {:.6f}, total {:.6f} Hartree\n", channels_only,
             data.total_solvation_energy);
  REQUIRE(data.total_solvation_energy != Catch::Approx(channels_only));
}

TEST_CASE("Solvent specs parse names and mixtures", "[solvent][cosmors]") {
  using occ::driver::SolventSpec;

  SECTION("A bare name is a pure solvent") {
    auto spec = SolventSpec::parse("water");
    REQUIRE_FALSE(spec.is_mixture());
    REQUIRE(spec.single() == "water");
    REQUIRE(spec.to_string() == "water");
    REQUIRE(spec.filename_tag() == "water");
  }

  SECTION("Fractions are normalised") {
    auto spec = SolventSpec::parse("water:3,ethanol:1");
    REQUIRE(spec.is_mixture());
    REQUIRE(spec.components.size() == 2);
    REQUIRE(spec.mole_fractions(0) == Catch::Approx(0.75));
    REQUIRE(spec.mole_fractions(1) == Catch::Approx(0.25));
    REQUIRE(spec.mole_fractions.sum() == Catch::Approx(1.0));
    // Colons and commas are not portable in filenames.
    REQUIRE(spec.filename_tag() == "water0.75-ethanol0.25");
    REQUIRE_THROWS(spec.single());
  }

  SECTION("Whitespace is tolerated") {
    auto spec = SolventSpec::parse(" water : 0.7 , ethanol : 0.3 ");
    REQUIRE(spec.components[0] == "water");
    REQUIRE(spec.components[1] == "ethanol");
  }

  SECTION("A mixture without fractions is rejected") {
    REQUIRE_THROWS_WITH(SolventSpec::parse("water,ethanol"),
                        Catch::Matchers::ContainsSubstring("mole fraction"));
  }
}

TEST_CASE("Solvation model selection and its guard rails",
          "[solvent][cosmors]") {
  using namespace occ::driver;

  REQUIRE(parse_solvation_model("smd") == SolvationModelKind::Smd);
  REQUIRE(parse_solvation_model("cosmo-rs") == SolvationModelKind::CosmoRS);
  REQUIRE(parse_solvation_model("cosmors") == SolvationModelKind::CosmoRS);
  REQUIRE(parse_solvation_model("cosmors") == SolvationModelKind::CosmoRS);
  REQUIRE(parse_solvation_model("none") == SolvationModelKind::None);
  REQUIRE(parse_solvation_model("gas") == SolvationModelKind::None);
  REQUIRE_THROWS(parse_solvation_model("nonsense"));

  REQUIRE(solvation_model_name(SolvationModelKind::CosmoRS) == "cosmo-rs");

  CGSolvationSettings settings;
  auto smd = make_cg_solvation_model(SolvationModelKind::Smd, settings);
  auto cosmors = make_cg_solvation_model(SolvationModelKind::CosmoRS, settings);

  // SMD polarises against the real dielectric; openCOSMO-RS's reference is
  // the ideal conductor, so it must not be asked for solvated wavefunctions.
  REQUIRE(smd->supports_solvated_wavefunctions());
  REQUIRE_FALSE(cosmors->supports_solvated_wavefunctions());

  auto mixture = SolventSpec::parse("water:0.7,ethanol:0.3");
  REQUIRE_THROWS_WITH(smd->validate(mixture),
                      Catch::Matchers::ContainsSubstring("mixtures"));
  REQUIRE_NOTHROW(cosmors->validate(mixture));
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
