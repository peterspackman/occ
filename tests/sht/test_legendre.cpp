#include "catch.hpp"
#include <occ/core/util.h>
#include <occ/sht/legendre.h>

TEST_CASE("Evaluate batch", "[sht]") {

    occ::sht::AssocLegendreP plm(4);
    occ::Vec expected((4 + 1) * (4 + 2) / 2);
    occ::Vec result((4 + 1) * (4 + 2) / 2);

    expected << 0.28209479177387814, 0.24430125595145993, -0.07884789131313,
        -0.326529291016351, -0.24462907724141, 0.2992067103010745,
        0.33452327177864455, 0.06997056236064664, -0.25606603842001846,
        0.2897056515173922, 0.3832445536624809, 0.18816934037548755,
        0.27099482274755193, 0.4064922341213279, 0.24892463950030275;

    plm.evaluate_batch(0.5, result);

    REQUIRE(occ::util::all_close(expected, result));
}
