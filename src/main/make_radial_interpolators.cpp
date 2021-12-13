#include <fmt/core.h>
#include <occ/core/eigenp.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/element.h>
#include <occ/slater/thakkar.h>

int main(int argc, char *argv[]) {
    occ::thakkar::write_json_file();
    constexpr int num_points = 4096;
    auto basis = occ::slater::load_slaterbasis("thakkar");
    occ::Vec domain = occ::Vec::LinSpaced(num_points, -4, 3);
    domain = Eigen::exp(domain.array());
    auto rho = occ::MatRM(103, num_points);
    auto grad_rho = occ::MatRM(103, num_points);
    for (size_t i = 1; i <= 103; i++) {
        auto el = occ::core::Element(i);
        auto b = basis[el.symbol()];
        rho.row(i - 1) = b.rho(domain);
        grad_rho.row(i - 1) = b.grad_rho(domain);
        fmt::print("Atomic number {}\n", i);
    }
    enpy::save_npy("domain.npy", domain);
    enpy::save_npy("rho.npy", rho);
    enpy::save_npy("grad_rho.npy", grad_rho);

    return 0;
}
