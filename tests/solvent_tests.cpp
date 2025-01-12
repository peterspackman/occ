#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/eeq.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/io/xyz.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/solvent/surface.h>

using occ::format_matrix;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;
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
