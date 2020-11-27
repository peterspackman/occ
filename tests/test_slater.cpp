#include "linear_algebra.h"
#include "catch.hpp"
#include <fmt/ostream.h>
#include "timings.h"
#include "thakkar.h"
#include "util.h"

using tonto::util::all_close;

TEST_CASE("H2/STO-3G") {
    using tonto::Vec;
    auto H = tonto::thakkar::basis_for_element(1);
    auto C = tonto::thakkar::basis_for_element(6);

    Vec r0(1);
    r0(0) = 0.0;
    fmt::print("H(0) = {}\n", H.rho(0.0));
    fmt::print("H([0.0]) = {}\n", H.rho(r0));
    Vec r = Vec::LinSpaced(4, 0.2, 2);

    Vec rho = H.rho(r);
    Vec grad_rho = H.grad_rho(r);
    fmt::print("r\n{}\nH.rho\n{}\n", r, rho);
    Vec grad_rho_fd = (H.rho(r.array() + 1e-8) - H.rho(r.array() - 1e-8)) / 2e-8;
    fmt::print("H.grad_rho:\n{}\n", grad_rho);
    fmt::print("H.grad_rho_fd:\n{}\n", grad_rho_fd);
    REQUIRE(all_close(grad_rho, grad_rho_fd, 1e-5, 1e-5));
    rho = C.rho(r);
    fmt::print("r\n{}\nC.rho\n{}\n", r, rho);
    grad_rho = C.grad_rho(r);
    grad_rho_fd = (C.rho(r.array() + 1e-8) - C.rho(r.array() - 1e-8)) / 2e-8;
    REQUIRE(all_close(grad_rho, grad_rho_fd, 1e-5, 1e-5));


    tonto::timing::StopWatch<1> sw;
    Vec rtest = Vec::LinSpaced(100000, 0.01, 5.0);
    Vec rho_test(rtest.rows());
    sw.start(0);
    rho_test = C.rho(rtest);
    sw.stop(0);
    fmt::print("Time for {} points vec: {}\n", rho_test.rows(), sw.read(0));
    sw.clear_all();
    sw.start(0);
    for(size_t i = 0; i < rho_test.rows(); i++)
    {
        rho_test(i) = C.rho(rtest(i));
    }
    sw.stop(0);
    fmt::print("Time for {} points loop: {}\n", rho_test.rows(), sw.read(0));
}
