#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/io/structure_format.h>
#include <nlohmann/json.hpp>

using Catch::Approx;

TEST_CASE("SiteMultipoles flat round-trip", "[structure_format]") {
    occ::io::SiteMultipoles m;
    m.charge = 0.2095;
    m.dipole = {0.0, 0.00356, -0.0276};
    m.quadrupole = {-0.5952, 0.0, 0.0, -0.00576, 0.0};
    m.octupole = {0.0, -0.1191, 0.2849, 0.0, 0.0, -0.1768, 0.0};
    m.hexadecapole = {0.2075, 0.0, 0.1453, 0.0, -0.0384, 0.0, 0.0, -0.0465, 0.0};

    REQUIRE(m.max_rank() == 4);

    auto flat = m.to_flat();
    REQUIRE(flat.size() == 25);
    REQUIRE(flat[0] == Approx(0.2095));

    auto m2 = occ::io::SiteMultipoles::from_flat(flat);
    REQUIRE(m2.charge == Approx(m.charge));
    REQUIRE(m2.dipole.size() == 3);
    REQUIRE(m2.quadrupole.size() == 5);
    REQUIRE(m2.octupole.size() == 7);
    REQUIRE(m2.hexadecapole.size() == 9);

    for (int i = 0; i < 3; ++i)
        REQUIRE(m2.dipole[i] == Approx(m.dipole[i]));
    for (int i = 0; i < 5; ++i)
        REQUIRE(m2.quadrupole[i] == Approx(m.quadrupole[i]));
}

TEST_CASE("StructureInput JSON round-trip", "[structure_format]") {
    occ::io::StructureInput si;
    si.title = "TEST";
    si.a = 5.0; si.b = 6.0; si.c = 7.0;
    si.alpha = 90.0; si.beta = 90.0; si.gamma = 90.0;

    occ::io::MoleculeType mt;
    mt.name = "test_mol";
    occ::io::MoleculeSite site;
    site.label = "C1";
    site.element = "C";
    site.type = "C_W3";
    site.position = {0.1, 0.2, 0.3};
    site.multipoles.charge = 0.5;
    site.multipoles.dipole = {0.1, 0.2, 0.3};
    mt.sites.push_back(site);
    si.molecule_types.push_back(mt);

    occ::io::SymmetryEntry se;
    se.op = "x, y, z";
    se.molecule = 0;
    si.symmetry.push_back(se);

    occ::io::IndependentMolecule im;
    im.type = "test_mol";
    im.translation = {0.5, 0.5, 0.5};
    im.orientation = {0.0, 0.0, 0.0};
    si.molecules.push_back(im);

    occ::io::BuckinghamPair bp;
    bp.types = {"C_W3", "C_W3"};
    bp.elements = {"C", "C"};
    bp.A = 2802.33;
    bp.rho = 0.2778;
    bp.C6 = 17.639;
    si.potentials.buckingham.push_back(bp);
    si.potentials.cutoff = 15.0;

    si.reference.total = -90.218;
    si.reference.components["buckingham"] = -16.937;

    // Serialize to JSON and back
    nlohmann::json j = si;
    auto si2 = j.get<occ::io::StructureInput>();

    REQUIRE(si2.title == "TEST");
    REQUIRE(si2.a == Approx(5.0));
    REQUIRE(si2.b == Approx(6.0));
    REQUIRE(si2.c == Approx(7.0));
    REQUIRE(si2.molecule_types.size() == 1);
    REQUIRE(si2.molecule_types[0].name == "test_mol");
    REQUIRE(si2.molecule_types[0].sites.size() == 1);
    REQUIRE(si2.molecule_types[0].sites[0].element == "C");
    REQUIRE(si2.molecule_types[0].sites[0].type == "C_W3");
    REQUIRE(si2.molecule_types[0].sites[0].multipoles.charge == Approx(0.5));
    REQUIRE(si2.molecule_types[0].sites[0].multipoles.dipole.size() == 3);
    REQUIRE(si2.symmetry.size() == 1);
    REQUIRE(si2.symmetry[0].op == "x, y, z");
    REQUIRE(si2.molecules.size() == 1);
    REQUIRE(si2.molecules[0].type == "test_mol");
    REQUIRE(si2.molecules[0].translation[0] == Approx(0.5));
    REQUIRE(si2.potentials.buckingham.size() == 1);
    REQUIRE(si2.potentials.buckingham[0].A == Approx(2802.33));
    REQUIRE(si2.potentials.buckingham[0].elements[0] == "C");
    REQUIRE(si2.potentials.buckingham[0].elements[1] == "C");
    REQUIRE(si2.reference.total == Approx(-90.218));
    REQUIRE(si2.reference.components.at("buckingham") == Approx(-16.937));
}
