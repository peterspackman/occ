#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/numpy.h>
#include <occ/slater/slaterbasis.h>

int main(int argc, char *argv[]) {
  constexpr int num_points = 4096;
  auto basis = occ::slater::load_slaterbasis("thakkar");
  occ::Vec domain = occ::Vec::LinSpaced(num_points, 0.04, 400.0).array().sqrt();
  auto rho = occ::MatRM(103, num_points);
  auto grad_rho = occ::MatRM(103, num_points);
  for (size_t i = 1; i <= 103; i++) {
    auto el = occ::core::Element(i);
    auto b = basis[el.symbol()];
    rho.row(i - 1) = b.rho(domain);
    grad_rho.row(i - 1) = b.grad_rho(domain);
    fmt::print("Atomic number {}\n", i);
  }
  occ::core::numpy::save_npy("domain.npy", domain);
  occ::core::numpy::save_npy("rho.npy", rho);
  occ::core::numpy::save_npy("grad_rho.npy", grad_rho);

  return 0;
}
