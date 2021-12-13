#include <occ/core/element.h>
#include "catch.hpp"

TEST_CASE("Element constructor", "[element]")
{
    using occ::core::Element;
    REQUIRE(Element("H").symbol() == "H");
    REQUIRE(Element("He").symbol() == "He");
    REQUIRE(Element("He1").symbol() == "He");
    REQUIRE(Element(6).name() == "carbon");
    REQUIRE(Element("Ne") > Element("H"));
    REQUIRE(Element("NA").symbol() == "N");
    REQUIRE(Element("Na").symbol() == "Na");
}
