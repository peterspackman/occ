#include <array>
#include <catch2/catch_test_macros.hpp>
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/slater/slaterbasis.h>

using occ::format_matrix;
using occ::util::all_close;

std::array<double, 38> domain_data{
    0.000000, 0.020000, 0.040000, 0.060000, 0.080000, 0.100000, 0.120000,
    0.140000, 0.160000, 0.180000, 0.200000, 0.220000, 0.240000, 0.260000,
    0.280000, 0.300000, 0.320000, 0.340000, 0.360000, 0.380000, 0.400000,
    0.420000, 0.440000, 0.460000, 0.480000, 0.500000, 0.520000, 0.540000,
    0.560000, 0.580000, 0.600000, 0.620000, 0.640000, 0.660000, 0.680000,
    0.700000, 0.720000, 0.740000};

TEST_CASE("Grad and rho consistency") {
  using occ::Vec;
  auto basis = occ::slater::load_slaterbasis("thakkar");
  auto H = basis["H"];
  auto C = basis["C"];

  Vec r0(1);
  r0(0) = 0.0;
  fmt::print("H(0) = {}\n", H.rho(0.0));
  fmt::print("H([0.0]) = {}\n", format_matrix(H.rho(r0)));
  Vec r = Vec::LinSpaced(4, 0.2, 2);

  Vec rho = H.rho(r);
  Vec grad_rho = H.grad_rho(r);
  fmt::print("r\n{}\nH.rho\n{}\n", format_matrix(r), format_matrix(rho));
  Vec grad_rho_fd = (H.rho(r.array() + 1e-8) - H.rho(r.array() - 1e-8)) / 2e-8;
  fmt::print("H.grad_rho:\n{}\n", format_matrix(grad_rho));
  fmt::print("H.grad_rho_fd:\n{}\n", format_matrix(grad_rho_fd));
  REQUIRE(all_close(grad_rho, grad_rho_fd, 1e-5, 1e-5));
  rho = C.rho(r);
  fmt::print("r\n{}\nC.rho\n{}\n", format_matrix(r), format_matrix(rho));
  grad_rho = C.grad_rho(r);
  grad_rho_fd = (C.rho(r.array() + 1e-8) - C.rho(r.array() - 1e-8)) / 2e-8;
  REQUIRE(all_close(grad_rho, grad_rho_fd, 1e-5, 1e-5));
}

TEST_CASE("Vector vs. repeated function call") {
  auto basis = occ::slater::load_slaterbasis("thakkar");
  auto Ag = basis["Ag"];
  using occ::Vec;

  occ::timing::StopWatch<1> sw;
  Vec rtest = Vec::LinSpaced(100000, 0.01, 5.0);
  Vec rho_vec(rtest.rows()), rho_func(rtest.rows());
  sw.start(0);
  rho_vec = Ag.rho(rtest);
  sw.stop(0);
  fmt::print("Time for {} points vec: {}\n", rho_vec.rows(), sw.read(0));
  sw.clear_all();
  sw.start(0);
  for (size_t i = 0; i < rtest.rows(); i++) {
    rho_func(i) = Ag.rho(rtest(i));
  }
  sw.stop(0);
  fmt::print("Time for {} points loop: {}\n", rho_func.rows(), sw.read(0));
  REQUIRE(all_close(rho_func, rho_vec));
}

TEST_CASE("H atom slater basis density") {
  using occ::Vec;
  auto basis = occ::slater::load_slaterbasis("thakkar");
  auto H = basis["H"];
  Vec expected(38);
  Eigen::Map<Vec, 0> domain(domain_data.data(), 38);
  expected << 0.606897, 0.577530, 0.549583, 0.522989, 0.497681, 0.473598,
      0.450681, 0.428872, 0.408119, 0.388370, 0.369577, 0.351693, 0.334675,
      0.318480, 0.303069, 0.288403, 0.274447, 0.261167, 0.248529, 0.236503,
      0.225058, 0.214168, 0.203804, 0.193942, 0.184557, 0.175626, 0.167128,
      0.159041, 0.151345, 0.144021, 0.137052, 0.130420, 0.124109, 0.118103,
      0.112388, 0.106950, 0.101774, 0.096850;

  auto rho = H.rho(domain);
  fmt::print("Diff:\n{}\n", format_matrix(rho - expected));

  REQUIRE(all_close(rho, expected, 1e-5, 1e-5));
}

TEST_CASE("Ag atom slater basis density") {
  using occ::Vec;
  auto basis = occ::slater::load_slaterbasis("thakkar");
  auto Ag = basis["Ag"];
  Vec expected(38);
  Eigen::Map<Vec, 0> domain(domain_data.data(), 38);

  expected << 72806.532439, 11476.811931, 2564.204461, 1245.386938, 865.463781,
      597.159964, 392.515352, 255.349478, 173.017297, 127.076663, 102.004228,
      87.439613, 77.454621, 69.117356, 61.243252, 53.538364, 46.087621,
      39.082999, 32.698224, 27.044289, 22.164863, 18.048358, 14.644541,
      11.880118, 9.671165, 7.932082, 6.581441, 5.545408, 4.759384, 4.168430,
      3.726911, 3.397707, 3.151192, 2.964152, 2.818728, 2.701448, 2.602363,
      2.514311;

  auto rho = Ag.rho(domain);

  fmt::print("Ag errors\n{}\n", format_matrix(rho - expected));
  for (const auto &sh : Ag.shells()) {
    fmt::print("c:\n{}\n", format_matrix(sh.c()));
    fmt::print("z:\n{}\n", format_matrix(sh.z(), "{}"));
    fmt::print("n:\n{}\n", format_matrix(sh.n(), "{}"));
    fmt::print("occ:\n{}\n", format_matrix(sh.occupation(), "{}"));
  }
  REQUIRE(all_close(rho, expected, 1e-5, 1e-5));
}
