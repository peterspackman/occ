#include "tonto/geometry/marching_cubes.h"
#include "tonto/geometry/linear_hashed_marching_cubes.h"
#include "catch.hpp"
#include "fmt/core.h"
#include "fmt/os.h"

using tonto::geometry::mc::MarchingCubes;
using tonto::geometry::mc::LinearHashedMarchingCubes;

struct sphere
{
    float origin[3] = {0.5, 0.5, 0.5};
    float radius = 0.4;
    float operator()(float x, float y, float z) const
    {
        float x1 = x - origin[0], y1 = y - origin[1], z1 = z - origin[2];
        return sqrt(x1 * x1 + y1 * y1 + z1 * z1) - radius * radius;
    }

};

struct torus
{
    const float r1 = 0.25f;
    const float r2 = 0.1f;

    float sample(float x, float y, float z) const
    {
        float x1 = x - 0.5, y1 = x - 0.5, z1 = x - 0.5;
        float qx = sqrt(x1 * x1 + y1 * y1) - r1;
        float l = sqrt(qx * qx + z1 * z1);
        return l - r2;
    }

};

TEST_CASE("Marching cubes", "[geometry]")
{
    MarchingCubes m(64);
    sphere s;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    m.extract(s, vertices, indices);
    fmt::print("{} vertices, {} faces\n", vertices.size() / 3, indices.size() / 3);
    auto verts = fmt::output_file("verts.txt");
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 1], indices[i + 2]);
    }
}

TEST_CASE("Linear hashed marching cubes", "[geometry]")
{
    LinearHashedMarchingCubes m(7);
    sphere s;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    m.extract(s, vertices, indices);
    fmt::print("{} vertices, {} faces\n", vertices.size() / 3, indices.size() / 3);
    auto verts = fmt::output_file("verts_hashed.txt");
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1], vertices[i + 2]);
    }
    auto faces = fmt::output_file("faces_hashed.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 1], indices[i + 2]);
    }
}
