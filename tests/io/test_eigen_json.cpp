#include "catch.hpp"
#include <occ/io/eigen_json.h>
#include <occ/core/linear_algebra.h>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/util.h>

using nlohmann::json;

namespace test {
struct test_struct {
    occ::Vec vector;
    occ::Mat3 matrix3d;
    occ::RowVec3 rvec3;

    bool operator==(const test_struct &other) const {
	using occ::util::all_close;
	return all_close(vector, other.vector) &&
	    all_close(matrix3d, other.matrix3d) &&
	    all_close(rvec3, other.rvec3);
    }
};

void to_json(json &js, const test_struct &t) {
    js = {
	{"vector", t.vector},
	{"matrix3d", t.matrix3d},
	{"rvec3", t.rvec3}
    };
}

void from_json(const json &j, test_struct &t) {
    j.at("vector").get_to(t.vector);
    j.at("matrix3d").get_to(t.matrix3d);
    j.at("rvec3").get_to(t.rvec3);
}

}

TEST_CASE("eigen serialize/deserialize as part of struct", "[serialize,deserialize]")
{
    auto t = test::test_struct{occ::Vec::Zero(10), occ::Mat3::Identity(), occ::RowVec3::Zero()};
    nlohmann::json j = t;
    auto t2 = j.get<test::test_struct>();
    REQUIRE(t == t2);
}
