#include "catch.hpp"
#include <fmt/ostream.h>
#include "eigenp.h"
#include "util.h"
#include <cstdio>

using tonto::util::all_close;


TEST_CASE("write eigen array", "[numpy]")
{
    Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
    i(0, 4) = 4;
    const std::string filename = "identity.npy";
    enpy::save_npy(filename, i);
    enpy::NumpyArray arr = enpy::load_npy(filename);
    Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0], arr.shape[1]);
    REQUIRE(all_close(i, m));
    std::remove(filename.c_str());
}


TEST_CASE("write eigen array compressed", "[numpy]")
{
    Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
    i(0, 4) = 4;
    const std::string filename = "test.npz";
    enpy::save_npz(filename, "identity", i);
    enpy::NumpyArray arr = enpy::load_npz(filename, "identity");
    Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0], arr.shape[1]);
    REQUIRE(all_close(i, m));
    std::remove(filename.c_str());

}
