#include "catch.hpp"
#include "inputfile.h"
#include <sstream>
#include <fmt/ostream.h>

const char* contents = R"(%chk=h2o
#p HF/3-21G 6D 10f

this is a comment

0 1
O -0.0148150000 2.2222190000 0.0000000000
H 0.2124040000 2.7681990000 -0.7555250000
H 0.2124040000 2.7681990000 0.7555250000

)";


TEST_CASE("h2o gaussian input", "[read]")
{
    std::istringstream inp(contents);
    tonto::io::GaussianInputFile reader(inp);
    REQUIRE(reader.charge == 0);
}
