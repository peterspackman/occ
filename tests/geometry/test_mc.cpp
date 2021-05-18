#include "occ/geometry/marching_cubes.h"
#include "occ/geometry/linear_hashed_marching_cubes.h"
#include "catch.hpp"
#include "fmt/core.h"
#include "fmt/os.h"
#include <occ/core/timings.h>

using occ::geometry::mc::MarchingCubes;
using occ::geometry::mc::LinearHashedMarchingCubes;

struct sphere
{
    float origin[3] = {0.5, 0.5, 0.5};
    float radius = 0.4;
    mutable int num_calls = 0;
    float operator()(float x, float y, float z) const
    {
        num_calls++;
        float x1 = x - origin[0], y1 = y - origin[1], z1 = z - origin[2];
        return sqrt(x1 * x1 + y1 * y1 + z1 * z1) - radius;
    }

    std::array<float, 3> normal(float x, float y, float z) const
    {
        std::array<float, 3> res{x - origin[0], y - origin[1], z - origin[2]};
        float norm = sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
        for(auto & x: res) x /= norm;
        return res;
    }
};

struct torus
{
    const float r1 = 0.25f;
    const float r2 = 0.1f;

    mutable int num_calls = 0;
    float operator()(float x, float y, float z) const
    {
        num_calls++;
        float x1 = x - 0.5, y1 = y - 0.5, z1 = z - 0.5;
        float qx = sqrt(x1 * x1 + y1 * y1) - r1;
        float l = sqrt(qx * qx + z1 * z1);
        return l - r2;
    }

};

TEST_CASE("Marching cubes", "[geometry]")
{
    MarchingCubes m(128);
    torus s;
    occ::timing::StopWatch<1> sw;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    sw.start(0);
    m.extract(s, vertices, indices);
    sw.stop(0);
    fmt::print("{} vertices, {} faces in {}\n", vertices.size() / 3, indices.size() / 3, sw.read(0));
    auto verts = fmt::output_file("verts.txt");
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2], indices[i + 1]);
    }
    fmt::print("{} function calls\n", s.num_calls);
}

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
    fmt::print("{} vertices, {} faces in {}\n", vertices.size() / 3, indices.size() / 3, sw.read(0));
    auto verts = fmt::output_file("verts_hashed.txt");
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces_hashed.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2], indices[i + 1]);
    }
    fmt::print("{} function calls\n", s.num_calls);
}
