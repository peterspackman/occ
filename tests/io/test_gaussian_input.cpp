#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/io/inputfile.h>
#include <sstream>

const char *contents = R""""(%chk=h2o
#p HF/3-21G 6D 10f

this is a comment

0 1
O -0.0148150000  2.2222190000  0.0000000000
H  0.2124040000  2.7681990000 -0.7555250000
H  0.2124040000  2.7681990000  0.7555250000

)"""";

TEST_CASE("h2o gaussian input", "[read]") {
    std::istringstream inp(contents);
    occ::io::GaussianInputFile reader(inp);
    REQUIRE(reader.charge == 0);
    REQUIRE(reader.method == "hf");
    REQUIRE(reader.basis_name == "3-21g");

    REQUIRE(reader.atomic_positions.size() == 3);
    REQUIRE(reader.atomic_numbers.size() == 3);
    REQUIRE(reader.atomic_numbers[2] == 1);
    REQUIRE(reader.spinorbital_kind() == occ::qm::SpinorbitalKind::Restricted);
}
