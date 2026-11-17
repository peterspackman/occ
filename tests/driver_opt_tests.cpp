#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/molecule.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/neb.h>
#include <occ/driver/vibrational_analysis.h>
#include <occ/io/occ_input.h>
#include <occ/io/xyz.h>
#include <filesystem>
#include <fstream>

using Catch::Approx;

namespace {
struct DataDirGuard {
  DataDirGuard() { occ::set_data_directory(OCC_GFN2_DATA_DIR); }
};
DataDirGuard _data_guard;

// Runs a test in its own temporary directory, where the drivers write their
// trajectories, geometries and frequencies.
struct InTemporaryDirectory {
  std::filesystem::path previous, dir;
  explicit InTemporaryDirectory(const std::string &name)
      : previous(std::filesystem::current_path()),
        dir(std::filesystem::temp_directory_path() / name) {
    std::filesystem::create_directories(dir);
    std::filesystem::current_path(dir);
  }
  ~InTemporaryDirectory() { std::filesystem::current_path(previous); }
};

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
  InTemporaryDirectory scratch("occ_driver_freq_test");
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
  InTemporaryDirectory scratch("occ_driver_ts_test");
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

TEST_CASE("GFN2-xTB NEB from HCN to HNC refined to the transition state",
          "[driver][neb]") {
  // GFN2-xTB minima
  occ::IVec z(3);
  z << 6, 7, 1;
  occ::Mat3N hcn(3, 3), hnc(3, 3);
  hcn << 0.003608, 1.141206, -1.054814, 0.006936, -0.003179, 0.016243, 0.0,
      0.0, 0.0;
  hnc << 0.008570, 1.166961, 2.164469, -0.003264, 0.007216, 0.016048, 0.0,
      0.0, 0.0;
  InTemporaryDirectory scratch("occ_driver_neb_test");
  const auto &dir = scratch.dir;
  const auto product = (dir / "hnc.xyz").string();
  {
    std::ofstream out(product);
    out << occ::io::to_xyz_string(occ::core::Molecule(z, hnc));
  }
  occ::io::OccInput config;
  config.method.name = "gfn2";
  config.filename = (dir / "hcn.xyz").string();
  config.geometry.set_molecule(occ::core::Molecule(z, hcn));
  config.optimization.neb_product = product;
  config.optimization.neb_ts = true;
  const auto result = occ::driver::nudged_elastic_band(config);
  REQUIRE(result.converged);
  REQUIRE(result.transition_state);
  // the saddle point of the direct search above
  CHECK(result.transition_state->energy.total ==
        Approx(-5.3873733).margin(1e-6));
  const occ::Vec f = result.ts_modes.get_all_frequencies();
  CHECK(f.minCoeff() == Approx(-1426.0).margin(5.0));
}
