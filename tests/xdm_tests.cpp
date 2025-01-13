#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/xdm/xdm.h>

using occ::format_matrix;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;

constexpr std::array<double, 103> frepol{
    0.e0,      0.6668e0,  0.2051e0,  24.3300e0, 5.6000e0,  3.0300e0,  1.7600e0,
    1.1000e0,  0.8020e0,  0.5570e0,  0.3956e0,  24.1100e0, 10.6000e0, 6.8000e0,
    5.3800e0,  3.6300e0,  2.9000e0,  2.1800e0,  1.6411e0,  43.4000e0, 22.8000e0,
    17.8000e0, 14.6000e0, 12.4000e0, 11.6000e0, 9.4000e0,  8.4000e0,  7.5000e0,
    6.8000e0,  6.2000e0,  5.7500e0,  8.1200e0,  6.0700e0,  4.3100e0,  3.7700e0,
    3.0500e0,  2.4844e0,  47.3000e0, 27.6000e0, 22.7000e0, 17.9000e0, 15.7000e0,
    12.8000e0, 11.4000e0, 9.6000e0,  8.6000e0,  4.8000e0,  7.2000e0,  7.3600e0,
    10.2000e0, 7.7000e0,  6.6000e0,  5.5000e0,  5.3500e0,  4.0440e0,  59.4200e0,
    39.7000e0, 31.1000e0, 29.6000e0, 28.2000e0, 31.4000e0, 30.1000e0, 28.8000e0,
    27.7000e0, 23.5000e0, 25.5000e0, 24.5000e0, 23.6000e0, 22.7000e0, 21.8000e0,
    21.0000e0, 21.9000e0, 16.2000e0, 13.1000e0, 11.1000e0, 9.7000e0,  8.5000e0,
    7.6000e0,  6.5000e0,  5.8000e0,  5.0200e0,  7.6000e0,  6.8000e0,  7.4000e0,
    6.8000e0,  6.0000e0,  5.3000e0,  48.6000e0, 38.3000e0, 32.1000e0, 32.1000e0,
    25.4000e0, 24.9000e0, 24.8000e0, 24.5000e0, 23.3000e0, 23.0000e0, 22.7000e0,
    20.5000e0, 19.7000e0, 23.8000e0, 18.2000e0, 17.5000e0};

double xdm_polarizability(int n, double v, double vfree) {
  // this will be the equivalent polarizability to POSTG, which is the point
  // of comparison.
  double p = frepol[n] / (0.52917720859e0 * 0.52917720859e0 * 0.52917720859e0);
  return v * p / vfree;
}

TEST_CASE("Water XDM dispersion energy", "[xdm]") {

  Vec polarizabilities(3);

  std::vector<occ::core::Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };

  Mat moments(3, 3);
  moments << 4.4746785658e+00, 1.0884283711e+00, 1.0929537397e+00,
      2.8898799370e+01, 6.6743870561e+00, 6.6878435445e+00, 1.8355332889e+02,
      6.7438103019e+01, 6.7479436992e+01;

  Vec volume(3);
  volume << 1.8941883873e+01, 3.8999421841e+00, 3.9031603917e+00;

  Vec free_volume(3);
  free_volume << 2.3504820578e+01, 8.7017290471e+00, 8.7017290471e+00;

  for (int i = 0; i < 3; i++) {
    polarizabilities(i) =
        volume(i) *
        occ::core::Element(atoms[i].atomic_number).polarizability(false) /
        free_volume(i);

    polarizabilities(i) =
        xdm_polarizability(atoms[i].atomic_number, volume(i), free_volume(i));
  }

  double expected = -3.976524483082e-04;

  Mat expected_forces(3, 3);

  expected_forces << 2.248250121483e-07, 9.975579765949e-07,
      -1.222382988743e-06, 3.350812651483e-07, -8.612850877111e-07,
      5.262038225628e-07, -8.780610885086e-09, 1.877910301194e-08,
      -9.998492126856e-09;

  occ::xdm::XDM::Parameters params{0.7, 1.4};

  occ::xdm::XDMAtomList inp{atoms, polarizabilities, moments, volume,
                            free_volume};

  double energy;
  Mat3N forces;
  std::tie(energy, forces) = occ::xdm::xdm_dispersion_energy(inp, params);

  REQUIRE(energy == Catch::Approx(expected).margin(1e-8));
  fmt::print("Forces:\n{}\n", format_matrix(forces));
  fmt::print("Expected forces:\n{}\n", format_matrix(expected_forces));
  REQUIRE(occ::util::all_close(forces, expected_forces, 1e-8, 1e-8));
}
