#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <fmt/os.h>
#include <occ/core/timings.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/geometry/morton.h>

// Marching Cubes

using occ::geometry::mc::LinearHashedMarchingCubes;
using occ::geometry::mc::MarchingCubes;

struct sphere {
    float origin[3] = {0.5, 0.5, 0.5};
    float radius = 0.4;
    mutable int num_calls = 0;
    float operator()(float x, float y, float z) const {
        num_calls++;
        float x1 = x - origin[0], y1 = y - origin[1], z1 = z - origin[2];
        return sqrt(x1 * x1 + y1 * y1 + z1 * z1) - radius;
    }

    std::array<float, 3> normal(float x, float y, float z) const {
        std::array<float, 3> res{x - origin[0], y - origin[1], z - origin[2]};
        float norm = sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
        for (auto &x : res)
            x /= norm;
        return res;
    }
};

struct torus {
    const float r1 = 0.25f;
    const float r2 = 0.1f;

    mutable int num_calls = 0;
    float operator()(float x, float y, float z) const {
        num_calls++;
        float x1 = x - 0.5, y1 = y - 0.5, z1 = z - 0.5;
        float qx = sqrt(x1 * x1 + y1 * y1) - r1;
        float l = sqrt(qx * qx + z1 * z1);
        return l - r2;
    }
};

TEST_CASE("Marching cubes", "[geometry]") {
    MarchingCubes m(128);
    torus s;
    occ::timing::StopWatch<1> sw;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    sw.start(0);
    m.extract(s, vertices, indices);
    sw.stop(0);
    fmt::print("{} vertices, {} faces in {}\n", vertices.size() / 3,
               indices.size() / 3, sw.read(0));
    auto verts = fmt::output_file("verts.txt");
    for (size_t i = 0; i < vertices.size(); i += 3) {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i],
                    vertices[i + 1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces.txt");
    for (size_t i = 0; i < indices.size(); i += 3) {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2],
                    indices[i + 1]);
    }
    fmt::print("{} function calls\n", s.num_calls);
}

/*
TEST_CASE("Linear hashed marching cubes", "[geometry]")
{
    LinearHashedMarchingCubes m(7);
    torus s;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    occ::timing::StopWatch<1> sw;
    sw.start(0);
    m.extract(s, vertices, indices);
    sw.stop(0);
    fmt::print("{} vertices, {} faces in {}\n", vertices.size() / 3,
indices.size() / 3, sw.read(0)); auto verts =
fmt::output_file("verts_hashed.txt"); for(size_t i = 0; i < vertices.size(); i
+= 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i +
1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces_hashed.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2],
indices[i + 1]);
    }
    fmt::print("{} function calls\n", s.num_calls);
}
*/

// Morton Codes

using occ::geometry::MIndex;

TEST_CASE("Morton Index", "[geometry]") {
    MIndex m{1};
    MIndex c1 = m.child(7);
    MIndex c2 = c1.child(3);

    REQUIRE(m.level() == 0);
    REQUIRE(c1.size() == Catch::Approx(0.25));
    REQUIRE(c2.size() == Catch::Approx(0.125));
    auto center = c2.center();
    REQUIRE(center.x == Catch::Approx(0.875));
    REQUIRE(center.y == Catch::Approx(0.875));
    REQUIRE(center.z == Catch::Approx(0.625));

    REQUIRE(c2.primal(1, 3).code == 0x0);
    REQUIRE(c2.dual(1, 3).code == 0xb6db6db6db6db6db);
}