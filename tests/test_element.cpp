#include "element.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("Element constructor", "[element]")
{
    using craso::chem::Element;
    REQUIRE(Element("H").symbol() == "H");
    REQUIRE(Element(6).name() == "carbon");
    REQUIRE(Element("Ne") > Element("H"));
}
