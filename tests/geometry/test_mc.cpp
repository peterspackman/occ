#include "tonto/geometry/marching_cubes.h"
#include "catch.hpp"
#include "fmt/core.h"

using tonto::geometry::mc::MarchingCubes;

struct sphere
{
    const float r = 0.25;
    float sample(float x, float y, float z) const {
        return sqrt(x * x + y * y + z * z) - r;
    };
};

TEST_CASE("Morton Index", "[geometry]")
{
    MarchingCubes m(128);
    sphere s;
    std::vector<float> vertices;
    std::vector<unsigned> indices;
    m.extract(s, vertices, indices);
    fmt::print("{} vertices, {} faces\n", vertices.size() / 3, indices.size() / 3);
}
