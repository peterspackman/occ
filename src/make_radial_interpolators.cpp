#include "eigenp.h"
#include "thakkar.h"
#include <fmt/core.h>

int main(int argc, char *argv[])
{
    constexpr int num_points = 4096;
    tonto::Vec domain = tonto::Vec::LinSpaced(num_points, -4, 3);
    domain = Eigen::exp(domain.array());
    auto rho = tonto::MatRM(103, num_points);
    auto grad_rho = tonto::MatRM(103, num_points);
    for(size_t i = 1; i <= 103; i++)
    {
        auto b = tonto::thakkar::basis_for_element(i);
        rho.row(i - 1) = b.rho(domain);
        grad_rho.row(i - 1) = b.grad_rho(domain);
        fmt::print("Atomic number {}\n", i);
    }
    enpy::save_npy("domain.npy", domain);
    enpy::save_npy("rho.npy", rho);
    enpy::save_npy("grad_rho.npy", grad_rho);

    return 0;
}
