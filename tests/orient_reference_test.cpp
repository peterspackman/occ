#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/multipole_interactions.h>
#include <fmt/core.h>
#include <cmath>
#include <string>

#include "orient_reference_data.h"

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;

TEST_CASE("Orient multi-direction reference validation",
          "[mults][orient_reference]") {

    MultipoleInteractions::Config config;
    config.max_rank = 4;
    MultipoleInteractions mults(config);

    Vec3 pos1(0.0, 0.0, 0.0);

    int passed = 0;
    int failed = 0;

    for (int i = 0; i < NUM_ORIENT_REFERENCE_ENTRIES; i++) {
        const auto& entry = ORIENT_REFERENCE_DATA[i];

        Vec3 pos2(entry.pos2_x, entry.pos2_y, entry.pos2_z);

        Mult m1(4);
        m1.get_component(entry.site1) = 1.0;

        Mult m2(4);
        m2.get_component(entry.site2) = 1.0;

        double occ_energy = mults.compute_interaction_energy(m1, pos1, m2, pos2);

        std::string label = fmt::format("{} x {} dir{}",
                                        entry.site1, entry.site2,
                                        entry.direction_idx);

        bool match = std::abs(occ_energy - entry.energy_hartree) < 1e-6;
        if (match) {
            passed++;
        } else {
            failed++;
            UNSCOPED_INFO(fmt::format(
                "MISMATCH {}: OCC={:.10e} Orient={:.10e} diff={:.2e}",
                label, occ_energy, entry.energy_hartree,
                std::abs(occ_energy - entry.energy_hartree)));
        }

        REQUIRE(occ_energy == Approx(entry.energy_hartree).margin(1e-6));
    }

    UNSCOPED_INFO(fmt::format(
        "Summary: {} tested, {} passed, {} failed",
        NUM_ORIENT_REFERENCE_ENTRIES, passed, failed));
}
