#include "catch.hpp"
#include <fmt/ostream.h>
#include "eigenp.h"
#include "util.h"
using tonto::util::all_close;

TEST_CASE("write eigen array", "[numpy]")
{
    Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
    i(0, 4) = 4;
    enpy::save_npz("test.npz", "identity", i);
    enpy::NumpyArray arr = enpy::load_npz("test.npz", "identity");
    Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0], arr.shape[1]);
    REQUIRE(all_close(i, m));
}
