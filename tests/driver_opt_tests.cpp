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
