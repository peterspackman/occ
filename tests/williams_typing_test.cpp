#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <cmath>
#include <set>

using namespace occ::mults;
using Approx = Catch::Approx;

TEST_CASE("BuckinghamDE uses bonded Williams atom typing", "[mults][williams_typing]") {
    const std::string json_path =
        std::string(CMAKE_SOURCE_DIR) + "/tests/data/dmacrys/AXOSOW.json";

    auto input = read_dmacrys_json(json_path);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::BuckinghamDE, true, false);

    REQUIRE(calc.uses_williams_atom_typing());

    std::set<int> types;
    for (const auto& geom : calc.molecule_geometry()) {
        for (int code : geom.short_range_type_codes) {
            if (code > 0) types.insert(code);
        }
    }

    CHECK(types.count(501) == 1); // H_W1 present
    CHECK(types.count(531) == 1); // O_W1 present
    CHECK((types.count(511) + types.count(512) + types.count(513)) > 0); // some carbon type
}

TEST_CASE("Williams typed Buckingham parameters use DMACRYS-scale units", "[mults][williams_typing]") {
    const std::string json_path =
        std::string(CMAKE_SOURCE_DIR) + "/tests/data/dmacrys/AXOSOW.json";

    auto input = read_dmacrys_json(json_path);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::BuckinghamDE, true, false);

    const auto p = calc.get_buckingham_params_for_types(512, 501); // C_W3-H_W1
    const double expected_B = 0.5 * ((1.0 / 0.277778) + (1.0 / 0.280899));

    CHECK(p.A > 5.0e4);
    CHECK(p.C > 6.0e2);
    CHECK(p.B == Approx(expected_B).margin(1e-5));

    // Typed and element-only generic parameters should not be identical.
    const auto p_elem = calc.get_buckingham_params(6, 1);
    CHECK(std::fabs(p.A - p_elem.A) > 1.0);

    // Halogen typed terms should be available from Williams defaults.
    REQUIRE(calc.has_typed_buckingham_params(540, 513)); // F_01 - C_W2
    const auto p_fc = calc.get_buckingham_params_for_types(540, 513);
    const double expected_B_fc = 0.5 * ((1.0 / 0.240385) + (1.0 / 0.277778));
    CHECK(p_fc.B == Approx(expected_B_fc).margin(1e-5));
    CHECK(p_fc.A > 1.0e5);
}
