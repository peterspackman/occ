#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/dimer.h>
#include <occ/core/eigenp.h>
#include <occ/core/element.h>
#include <occ/core/graph.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/optimize.h>
#include <occ/core/point_group.h>
#include <occ/core/table.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>

/* Dimer tests */
using occ::core::Dimer;
using occ::core::MolecularPointGroup;
using occ::core::Molecule;
using occ::core::Multipole;
using occ::core::SymOp;
using occ::util::all_close;

TEST_CASE("Dimer constructor", "[dimer]") {
    occ::Mat3N pos(3, 3), pos2(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;

    Molecule m(nums, pos);

    auto masses = m.atomic_masses();
    occ::Vec3 expected_masses = {15.994, 1.00794, 1.00794};

    fmt::print("Atomic masses:\n{}\n\n", masses);
    REQUIRE(all_close(masses, expected_masses, 1e-3, 1e-3));
}

TEST_CASE("Dimer transform", "[dimer]") {
    occ::Mat3N pos(3, 3), pos2(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;
    pos2 = pos;
    pos2.topRows(1).array() *= -1;

    Molecule m(nums, pos), m2(nums, pos2);

    Dimer dim(m, m2);

    fmt::print("m1\n{}\n", m.positions());
    fmt::print("m2\n{}\n", m2.positions());

    auto transform = dim.symmetry_relation().value();

    fmt::print("Transform matrix:\n{}\n", transform);
    m.transform(transform, Molecule::Origin::Centroid);
    fmt::print("m1\n{}\n", m.positions());
    fmt::print("m2\n{}\n", m2.positions());
    REQUIRE(all_close(m.positions(), m2.positions(), 1e-5, 1e-5));
}

TEST_CASE("Dimer separations", "[dimer]") {
    occ::Mat3N pos(3, 3), pos2(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;
    pos2 = pos;
    pos2.topRows(1).array() *= -1;

    Molecule m(nums, pos), m2(nums, pos2);

    Dimer dim(m, m2);

    REQUIRE(dim.nearest_distance() == Approx(0.8605988136));
    REQUIRE(dim.centroid_distance() == Approx(1.8479851333));
    REQUIRE(dim.center_of_mass_distance() == Approx(2.5186418514));
}

// Element tests
TEST_CASE("Element constructor", "[element]") {
    using occ::core::Element;
    REQUIRE(Element("H").symbol() == "H");
    REQUIRE(Element("He").symbol() == "He");
    REQUIRE(Element("He1").symbol() == "He");
    REQUIRE(Element(6).name() == "carbon");
    REQUIRE(Element("Ne") > Element("H"));
    REQUIRE(Element("NA").symbol() == "N");
    REQUIRE(Element("Na").symbol() == "Na");
}

// Molecule

TEST_CASE("Molecule constructor", "[molecule]") {
    occ::Mat3N pos(3, 2);
    occ::IVec nums(2);
    nums << 1, 1;
    pos << -1.0, 1.0, 0.0, 0.0, 1.0, 1.0;
    Molecule m(nums, pos);
    REQUIRE(m.size() == 2);
}

TEST_CASE("Molecule atom properties", "[molecule]") {
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -0.10593856, 0.01878821, -1.93166418, 1.60017351,
        -0.02171049, 0.48664409, 0.07959806, 0.00986248;
    Molecule m(nums, pos);

    auto masses = m.atomic_masses();
    occ::Vec3 expected_masses = {15.994, 1.00794, 1.00794};

    fmt::print("Atomic masses:\n{}\n\n", masses);
    REQUIRE(all_close(masses, expected_masses, 1e-3, 1e-3));
}

TEST_CASE("Molecule centroids", "[molecule]") {
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;
    Molecule m(nums, pos);

    occ::Vec3 expected_centroid = {-0.92399257, 0.524611, 0.0023134};
    occ::Vec3 calc_centroid = m.centroid();
    fmt::print("Calculated centroid:\n{}\n\n", calc_centroid);
    REQUIRE(all_close(expected_centroid, calc_centroid, 1e-05, 1e-05));

    occ::Vec3 expected_com = {-1.25932093, -0.000102380208, 0.0160229578};
    occ::Vec3 calc_com = m.center_of_mass();
    fmt::print("Calculated center of mass:\n{}\n\n", calc_com);
    REQUIRE(all_close(expected_com, calc_com, 1e-05, 1e-05));
}

TEST_CASE("Molecule rotation & translation", "[molecule]") {
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
        0.07959806, 0.01878821, -0.02171049, 0.00986248;
    Molecule m(nums, pos);

    Eigen::Affine3d rotation360;
    rotation360 = Eigen::AngleAxis<double>(M_PI * 2, occ::Vec3(0, 1, 0));

    m.rotate(rotation360);
    REQUIRE(all_close(pos, m.positions()));

    Eigen::Affine3d rotation180;
    rotation180 = Eigen::AngleAxis<double>(M_PI, occ::Vec3(1, 0, 0));
    auto expected_pos = pos;
    expected_pos.bottomRows(2).array() *= -1;
    m.rotate(rotation180);
    fmt::print("Rot:\n{}\n", rotation180.linear());
    fmt::print("Expected:\n{}\nFound:\n{}\n", expected_pos, m.positions());
    REQUIRE(all_close(expected_pos, m.positions()));
}

// Multipole

TEST_CASE("Multipole constructor", "[multipole]") {
    auto c = Multipole<0>{};
    auto d = Multipole<1>{};
    auto q = Multipole<2>{};
    auto o = Multipole<3>{};
    fmt::print("Charge\n{}\n", c);
    fmt::print("Dipole\n{}\n", d);
    fmt::print("Quadrupole\n{}\n", q);
    fmt::print("Octupole\n{}\n", o);
    REQUIRE(c.charge() == 0.0);
}

TEST_CASE("Multipole addition", "[multipole]") {
    auto o = Multipole<3>{{1.0,  0.0, 0.0, 0.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
                           10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}};
    auto q = Multipole<2>{{1.0, 0.0, 0.0, 0.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}};
    auto sum_oq = o + q;
    fmt::print("Result\n{}\n", sum_oq);
    for (unsigned int i = 0; i < o.num_components; i++) {
        if (i < q.num_components)
            REQUIRE(sum_oq.components[i] ==
                    (o.components[i] + q.components[i]));
        else
            REQUIRE(sum_oq.components[i] == o.components[i]);
    }
}

// Numpy IO

TEST_CASE("write eigen array", "[numpy]") {
    Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
    i(0, 4) = 4;
    const std::string filename = "identity.npy";
    enpy::save_npy(filename, i);
    enpy::NumpyArray arr = enpy::load_npy(filename);
    Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0],
                                     arr.shape[1]);
    REQUIRE(all_close(i, m));
    std::remove(filename.c_str());
}

TEST_CASE("write eigen array compressed", "[numpy]") {
    Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
    i(0, 4) = 4;
    const std::string filename = "test.npz";
    enpy::save_npz(filename, "identity", i);
    enpy::NumpyArray arr = enpy::load_npz(filename, "identity");
    Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0],
                                     arr.shape[1]);
    REQUIRE(all_close(i, m));
    std::remove(filename.c_str());
}

// Optimize

double sx(double x) { return std::sin(x); }

double ax(double x) { return std::abs(x - 4); }

TEST_CASE("Brent") {
    auto x2 = [](double x) { return (x - 0.5) * (x - 0.5); };
    occ::opt::Brent brent(x2);
    double xmin = brent.xmin();
    fmt::print("Found minimum of (x - 0.5)^2 in {} evaluations: ({}, {})\n",
               brent.num_calls(), xmin, brent.f_xmin());
    REQUIRE(xmin == Approx(0.5));
    occ::opt::Brent brentsin(sx);
    xmin = brentsin.xmin();
    fmt::print("Found a minimum of sin(x) in {} evaluations: ({}, {})\n",
               brentsin.num_calls(), xmin, brentsin.f_xmin());
    REQUIRE(std::abs(xmin) == Approx(M_PI / 2));

    occ::opt::Brent ba(ax);
    xmin = ba.xmin();
    fmt::print("Found minimum of abs(x - 4) in {} evaluations ({}, {})\n",
               ba.num_calls(), xmin, ba.f_xmin());
}

// Molecular Point Group

TEST_CASE("Symop constructors", "[point_group]") {
    occ::Vec3 axis(0.0, 1.0, 0.0);
    double angle = 90.0;
    occ::Vec3 rotvec = angle * axis;
    auto s = SymOp::from_axis_angle(axis, angle);
    fmt::print("Transformation:\n{}\n", s.transformation);
    auto s2 = SymOp::from_rotation_vector(rotvec);
    fmt::print("Transformation:\n{}\n", s.transformation);
}

TEST_CASE("Water: C2v", "[point_group]") {
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -0.7021961, -0.0560603, 0.0099423, -1.0221932, 0.8467758, -0.0114887,
        0.2575211, 0.0421215, 0.0052190;

    Molecule m(nums, pos.transpose());

    MolecularPointGroup pg(m);

    fmt::print("Water group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::C2v);
    REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("Oxygen: Dooh", "[point_group]") {
    occ::Mat3N pos(3, 2);
    occ::IVec nums(2);
    nums << 8, 8;
    pos << -0.616, 0.616, 0, 0, 0, 0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("Oxygen group: {}\n", pg.point_group_string());
    for (const auto &sym : pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for (const auto &sym : pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }

    REQUIRE(pg.point_group() == occ::core::PointGroup::Dooh);
    REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("BF3: D3h", "[point_group]") {
    occ::Mat3N pos(3, 4);
    occ::IVec nums(4);
    nums << 5, 9, 9, 9;
    pos << 0.0, 0.0000, 0.8121, -0.8121, 0.0, -0.9377, 0.4689, 0.4689, 0.0,
        0.0000, 0.0000, 0.0000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("BF3 group: {}\n", pg.point_group_string());
    for (const auto &sym : pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for (const auto &sym : pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }
    REQUIRE(pg.point_group() == occ::core::PointGroup::D3h);
    REQUIRE(pg.symmetry_number() == 6);
}

TEST_CASE("Benzene: D6h", "[point_group]") {
    occ::Mat3N pos(3, 12);
    occ::IVec nums(12);
    nums << 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1;
    pos << -0.71308, -0.00904, 1.39904, 2.10308, 1.39904, -0.00904, -1.73551,
        -0.52025, 1.91025, 3.12551, 1.91025, -0.52025, 1.20378, -0.01566,
        -0.01566, 1.20378, 2.42321, 2.42321, 1.20378, -0.90111, -0.90111,
        1.20378, 3.30866, 3.30866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("Benzene group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::D6h);
}

TEST_CASE("CH4: Td", "[point_group]") {
    occ::Mat3N pos(3, 5);
    occ::IVec nums(5);
    nums << 6, 1, 1, 1, 1;
    pos << 0.0, 0.00, 1.026719, -0.513360, -0.513360, 0.0, 0.00, 0.000000,
        -0.889165, 0.889165, 0.0, 1.08, -0.363000, -0.363000, -0.363000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("CH4 group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::Td);
}

TEST_CASE("Cube: Oh", "[point_group]") {
    occ::Mat3N pos(3, 8);
    occ::IVec nums(8);

    nums.setConstant(6);
    pos << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);
    fmt::print("Cube group: {}\n", pg.point_group_string());
    for (const auto &sym : pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for (const auto &sym : pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }

    REQUIRE(pg.point_group() == occ::core::PointGroup::Oh);
}

TEST_CASE("CHFBrCl: C1", "[point_group]") {
    occ::Mat3N pos(3, 5);
    occ::IVec nums(5);

    nums << 6, 1, 9, 35, 17;
    pos << 0.000000, 0.000000, 1.026719, -0.513360, -0.513360, 0.000000,
        0.000000, 0.000000, -0.889165, 0.889165, 0.000000, 1.080000, -0.363000,
        -0.363000, -0.363000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);
    fmt::print("CHFBrCl group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::C1);
}

// Table TODO remove or update it

TEST_CASE("Table constructor", "[table]") {
    using occ::io::Table;

    Table t({"test", "columns"});
    std::vector<int> test{1, 2, 3};
    std::vector<std::string> columns{"this", "is", "a", "test"};
    t.set_column("test", test);
    t.set_column("columns", columns);
    t.print();
}

TEST_CASE("Table Eigen", "[table]") {
    using occ::io::Table;
    Eigen::MatrixXd r = Eigen::MatrixXd::Random(4, 3);
    Table t;

    t.set_column("random", r);
    t.print();
}

// Utils

TEST_CASE("all_close", "[util]") {
    using occ::util::all_close;

    occ::Mat x = occ::Mat::Identity(3, 3);
    occ::Mat x2 = occ::Mat::Identity(3, 3) * 2;
    REQUIRE(all_close(x, x));
    REQUIRE(!all_close(x, x2));
    REQUIRE(all_close(x, x2 * 0.5));
}

TEST_CASE("is_close", "[util]") {
    using occ::util::is_close;
    double x1 = 1e-6;
    double x2 = 2e-6;
    REQUIRE(!is_close(x1, x2));
    REQUIRE(is_close(x1, x2 / 2));
    REQUIRE(is_close(1e-17, 1e-18));
}

TEST_CASE("is_even, is_odd", "[util]") {
    using occ::util::is_even, occ::util::is_odd;
    REQUIRE(is_even(2));
    REQUIRE(is_odd(1));
    REQUIRE(is_even(0));
    REQUIRE(is_odd(1312321413));
    REQUIRE(!is_odd(1312421412));
    REQUIRE(!is_even(1312421411));
}

TEST_CASE("smallest_common_factor", "[util]") {
    using occ::util::smallest_common_factor;
    REQUIRE(smallest_common_factor(1, 3) == 1);
    REQUIRE(smallest_common_factor(-1, 3) == 1);
    REQUIRE(smallest_common_factor(-2, 4) == 2);
    REQUIRE(smallest_common_factor(-4, 12) == 4);
    REQUIRE(smallest_common_factor(12, 12) == 12);
    REQUIRE(smallest_common_factor(0, 12) == 12);
}

TEST_CASE("human_readable_size", "[util]") {
    using occ::util::human_readable_size;

    REQUIRE(human_readable_size(1024, "B") == "1.00KiB");
    REQUIRE(human_readable_size(1024 * 1024, "B") == "1.00MiB");
    REQUIRE(human_readable_size(1024 * 1024 * 1024, "B") == "1.00GiB");
    REQUIRE(human_readable_size(0, "B") == "0.00B");
}

TEST_CASE("graph traversal", "[graph]") {
    using Graph = occ::core::graph::Graph<int, int>;
    using vertex_desc = Graph::VertexDescriptor;
    Graph graph;

    std::vector<vertex_desc> descriptors;

    for (int i = 0; i < 10; i++) {
        descriptors.push_back(graph.add_vertex(i));
    }

    auto e1 = graph.add_edge(descriptors[0], descriptors[1], 1);
    auto e2 = graph.add_edge(descriptors[1], descriptors[2], 1);
    auto e3 = graph.add_edge(descriptors[2], descriptors[3], 1);
    auto e4 = graph.add_edge(descriptors[4], descriptors[5], 1);
    auto e5 = graph.add_edge(descriptors[4], descriptors[6], 1);
    auto e6 = graph.add_edge(descriptors[4], descriptors[7], 1);

    auto f = [](const vertex_desc &v) { fmt::print("vertex = {}\n", v); };

    fmt::print("Depth first:\n");
    graph.depth_first_traversal(descriptors[0], f);
    fmt::print("Breadth first:\n");
    graph.breadth_first_traversal(descriptors[0], f);
    auto components = graph.connected_components();
    REQUIRE(components[descriptors[0]] == components[descriptors[3]]);
    REQUIRE(components[descriptors[0]] != components[descriptors[4]]);
    REQUIRE(components[descriptors[4]] == components[descriptors[6]]);
    REQUIRE(components[descriptors[4]] == components[descriptors[5]]);
}
