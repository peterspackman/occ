#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/molecule.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/vibrational_analysis.h>
#include <occ/io/occ_input.h>

using Catch::Approx;

namespace {
struct DataDirGuard {
  DataDirGuard() { occ::set_data_directory(OCC_GFN2_DATA_DIR); }
};
DataDirGuard _data_guard;

occ::core::Molecule water() {
  occ::IVec z(3);
  z << 8, 1, 1;
  occ::Mat3N pos(3, 3);
  pos.col(0) << 0.0, 0.0, 0.1173;
  pos.col(1) << 0.0, 0.7572, -0.4692;
  pos.col(2) << 0.0, -0.7572, -0.4692;
  return occ::core::Molecule(z, pos);
}
} // namespace

TEST_CASE("GFN2-xTB frequencies after an optimisation", "[driver][freq]") {
  occ::io::OccInput config;
  config.method.name = "gfn2";
  config.geometry.set_molecule(water());
  auto [wfn, modes] =
      occ::driver::geometry_optimization_with_frequencies(config, true);
  const occ::Vec f = modes.get_all_frequencies();
  REQUIRE(f.size() == 9);
  // xtb 6.7 --hess at the GFN2 minimum: 1539.52, 3643.04, 3651.28 cm-1
  CHECK(f(6) == Approx(1539.5).margin(2.0));
  CHECK(f(7) == Approx(3643.0).margin(2.0));
  CHECK(f(8) == Approx(3651.3).margin(2.0));
}

TEST_CASE("GFN2-xTB transition state of HCN to HNC", "[driver][ts]") {
  occ::IVec z(3);
  z << 6, 7, 1;
  occ::Mat3N pos(3, 3);
  pos.col(0) << 0.0, 0.0, 0.0;
  pos.col(1) << 1.18, 0.0, 0.0;
  pos.col(2) << 0.45, 1.12, 0.0;
  occ::io::OccInput config;
  config.method.name = "gfn2";
  config.geometry.set_molecule(occ::core::Molecule(z, pos));
  config.optimization.transition_state = true;
  auto [wfn, modes] =
      occ::driver::geometry_optimization_with_frequencies(config, true);
  // ORCA 6.1 OptTS with xtb 6.7: E = -5.3873735 Ha, 1426i cm-1
  CHECK(wfn.energy.total == Approx(-5.3873733).margin(1e-6));
  const occ::Vec f = modes.get_all_frequencies();
  int imaginary = 0;
  for (int i = 0; i < f.size(); i++)
    imaginary += f(i) < -50.0;
  CHECK(imaginary == 1);
  CHECK(f.minCoeff() == Approx(-1426.0).margin(5.0));
}
