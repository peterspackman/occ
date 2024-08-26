#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fmt/os.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/geometry/icosphere_mesh.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/geometry/math_utils.h>
#include <occ/geometry/morton.h>
#include <occ/geometry/quickhull.h>
#include <occ/geometry/wulff.h>
#include <occ/io/obj.h>
#include <random>

// Marching Cubes

using Catch::Matchers::WithinAbs;
using occ::geometry::mc::LinearHashedMarchingCubes;
using occ::geometry::mc::MarchingCubes;

struct sphere {
  occ::FVec3 origin{0.5, 0.5, 0.5};
  float radius = 0.4;
  mutable int num_calls = 0;
  float operator()(const occ::FVec3 &pos) const {
    num_calls++;
    return (pos - origin).norm() - radius;
  }

  occ::FVec3 normal(const occ::FVec3 &pos) const {
    return (pos - origin).normalized();
  }
};

struct torus {
  const float r1 = 0.25f;
  const float r2 = 0.1f;

  mutable int num_calls = 0;
  float operator()(const occ::FVec3 &pos) const {
    num_calls++;
    float x1 = pos.x() - 0.5, y1 = pos.y() - 0.5, z1 = pos.z() - 0.5;
    float qx = sqrt(x1 * x1 + y1 * y1) - r1;
    float l = sqrt(qx * qx + z1 * z1);
    return l - r2;
  }
};

TEST_CASE("Marching cubes 16x16x16 on torus function", "[geometry]") {
  MarchingCubes m(16);
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
    verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1],
                vertices[i + 2]);
  }
  auto faces = fmt::output_file("faces.txt");
  for (size_t i = 0; i < indices.size(); i += 3) {
    faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2],
                indices[i + 1]);
  }
  fmt::print("{} function calls\n", s.num_calls);
}

TEST_CASE("Marching cubes sphere uneven", "[geometry]") {
  MarchingCubes m(32, 8, 16);
  sphere s;
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
    verts.print("{:20.12f} {:20.12f} {:20.12f}\n", vertices[i], vertices[i + 1],
                vertices[i + 2]);
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

TEST_CASE("Morton Index constructor & children", "[geometry]") {
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

TEST_CASE("Wulff", "[geometry]") {
  using occ::geometry::WulffConstruction;
  using occ::util::all_close;

  occ::Mat3N pts(3, 6);
  occ::Mat3N expected(3, 6);
  occ::Vec3 vec(0.89956101, 0.42844002, 0.0850243);

  pts << 0.27324055, 0.74569843, 0.24143472, 0.81361223, 0.9912406, 0.42422966,
      0.97878276, 0.47166676, 0.00768781, 0.05601098, 0.45920777, 0.92948902,
      0.04862679, 0.48147977, 0.6230096, 0.90899989, 0.29847469, 0.6059014;

  expected << -0.50987868, 0.09806682, 0.31297152, 0.54571224, 0.09227554,
      -0.2119635, -0.31136949, -0.31136949, -0.35781536, -0.43226196,
      -0.13036801, -0.62666634, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  occ::Mat3N result = occ::geometry::project_to_plane(pts, vec);

  REQUIRE(all_close(result, expected, 1e-6, 1e-6));

  occ::Vec energies(80);

  energies << 0.295353, 0.183972, 0.043829, 0.096313, 0.19904, 0.655211,
      0.251524, 0.742869, 0.160817, 0.222195, 0.111382, 0.181492, 0.286699,
      0.862337, 0.295353, 0.417948, 0.27163, 0.792305, 0.140143, 0.889187,
      0.295353, 0.183972, 0.043829, 0.096313, 0.19904, 0.655211, 0.251524,
      0.742869, 0.160817, 0.222195, 0.111382, 0.181492, 0.286699, 0.862337,
      0.295353, 0.417948, 0.27163, 0.792305, 0.140143, 0.889187, 0.295353,
      0.183972, 0.043829, 0.096313, 0.19904, 0.655211, 0.251524, 0.742869,
      0.160817, 0.222195, 0.111382, 0.181492, 0.286699, 0.862337, 0.295353,
      0.417948, 0.27163, 0.792305, 0.140143, 0.889187, 0.295353, 0.183972,
      0.043829, 0.096313, 0.19904, 0.655211, 0.251524, 0.742869, 0.160817,
      0.222195, 0.111382, 0.181492, 0.286699, 0.862337, 0.295353, 0.417948,
      0.27163, 0.792305, 0.140143, 0.889187;

  occ::Mat3N directions(3, 80);
  directions << -5.65425835e-01, 0.00000000e+00, -8.99557779e-01,
      -8.84792469e-01, -4.36008872e-01, 0.00000000e+00, -7.64986674e-01,
      -7.55844006e-01, 0.00000000e+00, -3.37020775e-01, -3.29826443e-01,
      -3.02384710e-01, -8.95844746e-01, 0.00000000e+00, -5.65425835e-01,
      -5.61703856e-01, -5.55261730e-01, -8.31598595e-01, -5.44880422e-01,
      -5.18723282e-01, 5.65425835e-01, 0.00000000e+00, 8.99557779e-01,
      8.84792469e-01, 4.36008872e-01, 0.00000000e+00, 7.64986674e-01,
      7.55844006e-01, 0.00000000e+00, 3.37020775e-01, 3.29826443e-01,
      3.02384710e-01, 8.95844746e-01, 0.00000000e+00, 5.65425835e-01,
      5.61703856e-01, 5.55261730e-01, 8.31598595e-01, 5.44880422e-01,
      5.18723282e-01, -5.65425835e-01, 0.00000000e+00, -8.99557779e-01,
      -8.84792469e-01, -4.36008872e-01, 0.00000000e+00, -7.64986674e-01,
      -7.55844006e-01, 0.00000000e+00, -3.37020775e-01, -3.29826443e-01,
      -3.02384710e-01, -8.95844746e-01, 0.00000000e+00, -5.65425835e-01,
      -5.61703856e-01, -5.55261730e-01, -8.31598595e-01, -5.44880422e-01,
      -5.18723282e-01, 5.65425835e-01, 0.00000000e+00, 8.99557779e-01,
      8.84792469e-01, 4.36008872e-01, 0.00000000e+00, 7.64986674e-01,
      7.55844006e-01, 0.00000000e+00, 3.37020775e-01, 3.29826443e-01,
      3.02384710e-01, 8.95844746e-01, 0.00000000e+00, 5.65425835e-01,
      5.61703856e-01, 5.55261730e-01, 8.31598595e-01, 5.44880422e-01,
      5.18723282e-01, -7.77761281e-01, 0.00000000e+00, 5.50820277e-17,
      5.41779133e-17, -5.99744118e-01, -5.62000720e-01, -5.26131792e-01,
      -5.19843776e-01, -1.00000000e+00, -9.27165663e-01, -4.53686799e-01,
      -8.31879639e-01, -4.10754346e-01, -8.05423498e-01, -7.77761281e-01,
      -7.72641580e-01, 3.39999750e-17, -3.81296803e-01, 3.33643033e-17,
      -3.56760214e-01, 7.77761281e-01, 0.00000000e+00, -5.50820277e-17,
      -5.41779133e-17, 5.99744118e-01, 5.62000720e-01, 5.26131792e-01,
      5.19843776e-01, 1.00000000e+00, 9.27165663e-01, 4.53686799e-01,
      8.31879639e-01, 4.10754346e-01, 8.05423498e-01, 7.77761281e-01,
      7.72641580e-01, -3.39999750e-17, 3.81296803e-01, -3.33643033e-17,
      3.56760214e-01, 7.77761281e-01, 0.00000000e+00, 5.50820277e-17,
      5.41779133e-17, 5.99744118e-01, 5.62000720e-01, 5.26131792e-01,
      5.19843776e-01, 1.00000000e+00, 9.27165663e-01, 4.53686799e-01,
      8.31879639e-01, 4.10754346e-01, 8.05423498e-01, 7.77761281e-01,
      7.72641580e-01, 3.39999750e-17, 3.81296803e-01, 3.33643033e-17,
      3.56760214e-01, -7.77761281e-01, 0.00000000e+00, -5.50820277e-17,
      -5.41779133e-17, -5.99744118e-01, -5.62000720e-01, -5.26131792e-01,
      -5.19843776e-01, -1.00000000e+00, -9.27165663e-01, -4.53686799e-01,
      -8.31879639e-01, -4.10754346e-01, -8.05423498e-01, -7.77761281e-01,
      -7.72641580e-01, -3.39999750e-17, -3.81296803e-01, -3.33643033e-17,
      -3.56760214e-01, -2.74556034e-01, 1.00000000e+00, -4.36801788e-01,
      4.65985286e-01, 6.70971875e-01, 8.27136743e-01, -3.71457571e-01,
      3.98073218e-01, 9.78022064e-17, -1.63648495e-01, -8.27878497e-01,
      4.65338320e-01, 1.69537777e-01, 5.92699746e-01, -2.74556034e-01,
      2.95827260e-01, -8.31675665e-01, -4.03802582e-01, 8.38513760e-01,
      -7.76948072e-01, 2.74556034e-01, -1.00000000e+00, 4.36801788e-01,
      -4.65985286e-01, -6.70971875e-01, -8.27136743e-01, 3.71457571e-01,
      -3.98073218e-01, -9.78022064e-17, 1.63648495e-01, 8.27878497e-01,
      -4.65338320e-01, -1.69537777e-01, -5.92699746e-01, 2.74556034e-01,
      -2.95827260e-01, 8.31675665e-01, 4.03802582e-01, -8.38513760e-01,
      7.76948072e-01, -2.74556034e-01, 1.00000000e+00, -4.36801788e-01,
      4.65985286e-01, 6.70971875e-01, 8.27136743e-01, -3.71457571e-01,
      3.98073218e-01, -9.78022064e-17, -1.63648495e-01, -8.27878497e-01,
      4.65338320e-01, 1.69537777e-01, 5.92699746e-01, -2.74556034e-01,
      2.95827260e-01, -8.31675665e-01, -4.03802582e-01, 8.38513760e-01,
      -7.76948072e-01, 2.74556034e-01, -1.00000000e+00, 4.36801788e-01,
      -4.65985286e-01, -6.70971875e-01, -8.27136743e-01, 3.71457571e-01,
      -3.98073218e-01, 9.78022064e-17, 1.63648495e-01, 8.27878497e-01,
      -4.65338320e-01, -1.69537777e-01, -5.92699746e-01, 2.74556034e-01,
      -2.95827260e-01, 8.31675665e-01, 4.03802582e-01, -8.38513760e-01,
      7.76948072e-01;

  auto wulff = WulffConstruction(directions, energies);

  REQUIRE(wulff.vertices().cols() == 28);
  REQUIRE(wulff.triangles().cols() == 52);

  /*
  // an example of how to save the mesh to file
  occ::io::IsosurfaceMesh io_mesh;
  io_mesh.vertices = wulff.vertices().cast<float>();
  io_mesh.faces = wulff.triangles();
  occ::io::write_obj_file("wulff.obj", io_mesh, {});
  */
}

using namespace quickhull;
using FloatType = float;
using vec3 = Eigen::Matrix<FloatType, 3, 1>;

template <typename T>
static Eigen::Matrix<T, 3, Eigen::Dynamic>
createSphere(T radius, size_t M,
             Eigen::Matrix<T, 3, 1> offset = Eigen::Matrix<T, 3, 1>(0, 0, 0)) {
  const T pi = 3.14159f;
  std::vector<T> positions;
  for (size_t i = 0; i <= M; i++) {
    FloatType y = std::sin(pi / 2 + static_cast<FloatType>(i) / (M)*pi);
    FloatType r = std::cos(pi / 2 + static_cast<FloatType>(i) / (M)*pi);
    FloatType K =
        FloatType(1) -
        std::abs(static_cast<FloatType>(static_cast<FloatType>(i) - M / 2.0f)) /
            static_cast<FloatType>(M / 2.0f);
    const size_t pcount = static_cast<size_t>(1 + K * M + FloatType(1) / 2);
    for (size_t j = 0; j < pcount; j++) {
      FloatType x =
          pcount > 1 ? r * std::cos(static_cast<FloatType>(j) / pcount * pi * 2)
                     : 0;
      FloatType z =
          pcount > 1 ? r * std::sin(static_cast<FloatType>(j) / pcount * pi * 2)
                     : 0;
      positions.push_back(x + offset.x());
      positions.push_back(y + offset.y());
      positions.push_back(z + offset.z());
    }
  }
  Eigen::Map<Eigen::Matrix<T, 3, Eigen::Dynamic>> pc(positions.data(), 3,
                                                     positions.size() / 3);
  return pc;
}

TEST_CASE("Basic hull", "[qh]") {
  // Setup test env
  const size_t N = 200;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 200);
  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;

  // Test 1 : Put N points inside unit cube. Result mesh must have exactly 8
  // vertices because the convex hull is the unit cube.
  for (int i = 0; i < 8; i++) {
    pc.col(i) = Eigen::Matrix<FloatType, 3, 1>(i & 1 ? -1 : 1, i & 2 ? -1 : 1,
                                               i & 4 ? -1 : 1);
  }
  hull = qh.getConvexHull(pc, true);
  // 6 cube faces, 2 triangles per face, 3 indices per triangle
  constexpr size_t num_idxs = 3 * 2 * 6;
  REQUIRE(hull.indices().size() == num_idxs);

  // true if we reduced the vertices
  auto reduced = hull.reduced();
  REQUIRE(reduced.vertices().cols() == 8);
  REQUIRE(reduced.indices().size() == num_idxs);

  auto hull2 = hull;
  REQUIRE(hull2.vertices().cols() == hull.vertices().cols());
  REQUIRE(hull2.vertices()(0, 0) == hull.vertices()(0, 0));
  REQUIRE(hull2.indices().size() == hull.indices().size());
  auto hull3 = std::move(hull);
  REQUIRE(hull.indices().size() == 0);

  // Test 1.1 : Same test, but using the original indices.
  hull = qh.getConvexHull(pc, true);
  REQUIRE(hull.indices().size() == 3 * 2 * 6);
  REQUIRE(hull.vertices().cols() == pc.cols());
}

TEST_CASE("Sphere hull", "[qh]") {
  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;

  // Test 2 : random N points from the boundary of unit sphere. Result mesh
  // must have exactly N points.
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      createSphere<FloatType>(1.0f, 50);
  hull = qh.getConvexHull(pc, true);
  REQUIRE(pc.size() == hull.vertices().size());
  hull = qh.getConvexHull(pc, true);
  // Add every vertex twice. This should not affect final mesh
  auto hull_double = qh.getConvexHull(pc.replicate(1, 2), true);
  REQUIRE(hull_double.indices().size() == hull.indices().size());

  // Test 2.1 : Multiply x components of the unit sphere vectors by a huge
  // number => essentially we get a line
  const FloatType mul = 2 * 2 * 2;
  while (true) {
    pc.row(0).array() *= mul;
    hull = qh.getConvexHull(pc, true);
    if (hull.indices().size() == 12) {
      break;
    }
  }
}

TEST_CASE("0D hull", "[qh]") {

  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;
  // Test 3: 0D
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 101);
  pc.array() *= 0.000001f;
  pc.col(0).setConstant(2.0f);
  hull = qh.getConvexHull(pc, true);
  REQUIRE(hull.indices().size() >= 12);
}

TEST_CASE("Planar hull", "[qh]") {
  // Test 4: 2d degenerate case
  QuickHull<FloatType> qh;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc(3, 4);
  pc.col(0) =
      Eigen::Matrix<FloatType, 3, 1>(-3.000000f, -0.250000f, -0.800000f);
  pc.col(1) = Eigen::Matrix<FloatType, 3, 1>(-3.000000f, 0.250000f, -0.800000f);
  pc.col(2) = Eigen::Matrix<FloatType, 3, 1>(-3.125000f, 0.250000f, -0.750000);
  pc.col(3) = Eigen::Matrix<FloatType, 3, 1>(-3.125000f, -0.250000f, -0.750000);
  auto hull = qh.getConvexHull(pc, true);
  REQUIRE(hull.indices().size() == 12);
  // REQUIRE(hull.vertices().size() == 4);
}

TEST_CASE("Circle cylinder hull", "[qh]") {

  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;
  const size_t N = 200;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Zero(3, N);
  // Test 5: first a planar circle, then make a cylinder out of it
  for (size_t i = 0; i < N; i++) {
    const FloatType alpha = static_cast<FloatType>(i) / N * 2 * 3.14159f;
    pc.col(i) =
        Eigen::Matrix<FloatType, 3, 1>(std::cos(alpha), 0, std::sin(alpha));
  }
  hull = qh.getConvexHull(pc, true);
  hull.writeWaveformOBJ("circle.obj");

  REQUIRE(hull.vertices().size() == pc.size());
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc2 = pc.replicate(1, 2);
  pc2.block(1, N, 1, N).array() += 1.0f;
  hull = qh.getConvexHull(pc2, true);
  REQUIRE(hull.vertices().size() == pc2.size());
  hull.writeWaveformOBJ("test.obj");
  REQUIRE(hull.indices().size() / 3 == static_cast<size_t>(pc2.cols()) * 2 - 4);
}

TEST_CASE("Test 6", "[qh]") {

  Catch::Generators::RandomFloatingGenerator<FloatType> gen(
      0.0f, 2 * 3.1415f, Catch::Generators::Detail::getSeed());
  QuickHull<FloatType> qh;
  ConvexHull<FloatType> hull;
  const size_t N = 200;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Zero(3, N);

  // Test 6
  for (int x = 0;; x++) {
    pc = Eigen::Matrix<FloatType, 3, Eigen::Dynamic>(3, N);
    const FloatType l = 1;
    const FloatType r = l / (std::pow(10, x));
    for (size_t i = 0; i < N; i++) {
      vec3 p = vec3(1, 0, 0) * i * l / (N - 1);
      FloatType a = gen.get();
      vec3 d = vec3(0, std::sin(a), std::cos(a)) * r;
      pc.col(i) = p + d;
    }
    hull = qh.getConvexHull(pc, true);
    if (hull.indices().size() == 12) {
      break;
    }
  }
}

TEST_CASE("Normals", "[qh]") {
  QuickHull<FloatType> qh;
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc(3, 3);
  pc.col(0) = Eigen::Matrix<FloatType, 3, 1>(0, 0, 0);
  pc.col(1) = Eigen::Matrix<FloatType, 3, 1>(1, 0, 0);
  pc.col(2) = Eigen::Matrix<FloatType, 3, 1>(0, 1, 0);

  std::array<vec3, 2> normal;
  for (size_t i = 0; i < 2; i++) {
    const bool CCW = i;
    const auto hull = qh.getConvexHull(pc, CCW, false);
    const auto vertices = hull.vertices();
    const auto indices = hull.indices();
    // REQUIRE(vertices.size() == 3);
    REQUIRE(indices.size() >= 6);
    const vec3 triangle[3] = {vertices.col(indices[0]),
                              vertices.col(indices[1]),
                              vertices.col(indices[2])};
    normal[i] =
        mathutils::triangle_normal(triangle[0], triangle[1], triangle[2]);
  }
  const auto dot = normal[0].dot(normal[1]);
  REQUIRE(dot == Catch::Approx(-1));
}

TEST_CASE("Planes", "[qh]") {
  Eigen::Matrix<FloatType, 3, 1> N(1, 0, 0);
  Eigen::Matrix<FloatType, 3, 1> p(2, 0, 0);
  Plane<FloatType> P(N, p);
  auto dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(3, 0, 0), P);
  REQUIRE(dist == Catch::Approx(1));
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(1, 0, 0), P);
  REQUIRE(dist == Catch::Approx(-1));
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(1, 0, 0), P);
  REQUIRE(dist == Catch::Approx(-1));
  N = Eigen::Matrix<FloatType, 3, 1>(2, 0, 0);
  P = Plane<FloatType>(N, p);
  dist = mathutils::getSignedDistanceToPlane(
      Eigen::Matrix<FloatType, 3, 1>(6, 0, 0), P);
  REQUIRE(dist == Catch::Approx(8));
}

TEST_CASE("Vector3", "[qh]") {
  typedef Eigen::Matrix<FloatType, 3, 1> vec3;
  vec3 a(1, 0, 0);
  vec3 b(1, 0, 0);

  vec3 c = b * (a.dot(b) / b.squaredNorm());
  REQUIRE((c - a).norm() < 0.00001f);

  a = vec3(1, 1, 0);
  b = vec3(1, 3, 0);
  c = a * (b.dot(a) / a.squaredNorm());
  REQUIRE((c - vec3(2, 2, 0)).norm() < 0.00001f);
}

TEST_CASE("Half edge output", "[qh]") {
  QuickHull<FloatType> qh;

  // 8 corner vertices of a cube + tons of vertices inside.
  // Output should be a half edge mesh with 12 faces (6 cube faces with 2
  // triangles per face) and 36 half edges (3 half edges per face).
  Eigen::Matrix<FloatType, 3, Eigen::Dynamic> pc =
      Eigen::Matrix<FloatType, 3, Eigen::Dynamic>::Random(3, 1008);
  // between -1 , 1
  for (int h = 1000; h < 1008; h++) {
    pc.col(h) = Eigen::Matrix<FloatType, 3, 1>(h & 1 ? -2 : 2, h & 2 ? -2 : 2,
                                               h & 4 ? -2 : 2);
  }

  HalfEdgeMesh<FloatType, size_t> mesh =
      qh.getConvexHullAsMesh(pc.data(), pc.cols(), true);
  REQUIRE(mesh.m_faces.size() == 12);
  REQUIRE(mesh.m_halfEdges.size() == 36);
  // REQUIRE(mesh.m_vertices.size() == 8);

  // Verify that for each face f, f.halfedgeIndex equals
  // next(next(next(f.halfedgeIndex))).
  for (const auto &f : mesh.m_faces) {
    size_t next = mesh.m_halfEdges[f.m_halfEdgeIndex].m_next;
    next = mesh.m_halfEdges[next].m_next;
    next = mesh.m_halfEdges[next].m_next;
    REQUIRE(next == f.m_halfEdgeIndex);
  }
}

TEST_CASE("Sphere tests", "[qh]") {
  QuickHull<FloatType> qh;
  FloatType y = 1;
  for (;;) {
    auto pc = createSphere<FloatType>(1, 100,
                                      Eigen::Matrix<FloatType, 3, 1>(0, y, 0));
    auto hull = qh.getConvexHull(pc, true);
    y *= 15;
    y /= 10;
    if (hull.indices().size() == 12) {
      break;
    }
  }

  // Test worst case scenario: more and more points on the unit sphere. All
  // points should be part of the convex hull, as long as we can make epsilon
  // smaller without running out of numerical accuracy.
  size_t i = 1;
  FloatType eps = 0.002f;
  for (;;) {
    auto pc =
        createSphere<FloatType>(1, i, Eigen::Matrix<FloatType, 3, 1>(0, 0, 0));
    auto hull = qh.getConvexHull(pc, true, eps);
    if (qh.getDiagnostics().m_failedHorizonEdges) {
      // This should not happen
      REQUIRE(false);
      break;
    }
    if (pc.size() == hull.vertices().size()) {
      // Fine, all the points on unit sphere do belong to the convex mesh.
      i += 1;
    } else {
      eps *= 0.5f;
    }
    if (i == 100) {
      break;
    }
  }
}

float sphere_mean_curvature(float radius) { return 1.0f / radius; }

float sphere_gaussian_curvature(float radius) {
  return 1.0f / (radius * radius);
}

TEST_CASE("Curvature calculations for sphere", "[curvature]") {
  const float radius = 0.4f;
  const float tolerance = 1e-2f; // Tolerance for curvature comparisons
  sphere s;
  s.radius = radius;

  occ::geometry::mc::MarchingCubes mc(32, 32, 32); // Marching cubes resolution
  std::vector<float> vertices, normals, curvatures;
  std::vector<uint32_t> indices;

  mc.extract_with_curvature(s, vertices, indices, normals, curvatures);

  REQUIRE(vertices.size() > 1);
  REQUIRE(vertices.size() % 3 == 0);
  REQUIRE(normals.size() % 3 == 0);
  REQUIRE(curvatures.size() % 2 == 0);

  for (size_t i = 0; i < vertices.size(); i += 3) {
    float x = vertices[i];
    float y = vertices[i + 1];
    float z = vertices[i + 2];

    float mean_curvature = curvatures[i / 3 * 2];
    float gaussian_curvature = curvatures[i / 3 * 2 + 1];

    float ref_mean_curvature = sphere_mean_curvature(radius);
    float ref_gaussian_curvature = sphere_gaussian_curvature(radius);

    REQUIRE(mean_curvature ==
            Catch::Approx(ref_mean_curvature).epsilon(tolerance));
    REQUIRE(gaussian_curvature ==
            Catch::Approx(ref_gaussian_curvature).epsilon(tolerance));
  }
}

struct PairHash {
  size_t operator()(const std::pair<int, int> &p) const {
    return ankerl::unordered_dense::hash<int>{}(p.first) ^
           (ankerl::unordered_dense::hash<int>{}(p.second) << 1);
  }
};

TEST_CASE("IcosphereMesh", "[icosphere]") {
  using occ::geometry::IcosphereMesh;

  SECTION("Initial construction") {
    IcosphereMesh icosphere(0);

    REQUIRE(icosphere.vertices().cols() == 12);
    REQUIRE(icosphere.faces().cols() == 20);
  }

  SECTION("Vertex normalization") {
    IcosphereMesh icosphere(0);
    const auto &vertices = icosphere.vertices();

    for (int i = 0; i < vertices.cols(); ++i) {
      REQUIRE_THAT(vertices.col(i).norm(), WithinAbs(1.0, 1e-6));
    }
  }

  SECTION("Face indices are valid") {
    IcosphereMesh icosphere(0);
    const auto &faces = icosphere.faces();
    const auto &vertices = icosphere.vertices();

    for (int i = 0; i < faces.cols(); ++i) {
      REQUIRE(faces(0, i) >= 0);
      REQUIRE(faces(1, i) >= 0);
      REQUIRE(faces(2, i) >= 0);
      REQUIRE(faces(0, i) < vertices.cols());
      REQUIRE(faces(1, i) < vertices.cols());
      REQUIRE(faces(2, i) < vertices.cols());
    }
  }

  SECTION("Subdivision increases complexity") {
    for (size_t i = 0; i < 3; ++i) {
      IcosphereMesh icosphere(i);
      auto [num_vertices, num_faces] = IcosphereMesh::compute_sizes(i);

      REQUIRE(icosphere.vertices().cols() == num_vertices);
      REQUIRE(icosphere.faces().cols() == num_faces);

      if (i > 0) {
        auto [prev_vertices, prev_faces] = IcosphereMesh::compute_sizes(i - 1);
        REQUIRE(num_vertices > prev_vertices);
        REQUIRE(num_faces > prev_faces);
      }
    }
  }
  using occ::Vec3;

  SECTION("Midpoint calculation") {
    IcosphereMesh icosphere0(0);
    IcosphereMesh icosphere1(1);
    const auto &vertices = icosphere1.vertices();
    const auto &faces = icosphere0.faces();

    const double tolerance =
        1e-7; // Tightened tolerance based on normalized calculations

    ankerl::unordered_dense::set<std::pair<int, int>, PairHash> checked_edges;

    for (int i = 0; i < faces.cols(); ++i) {
      for (int j = 0; j < 3; ++j) {
        int v1 = faces(j, i);
        int v2 = faces((j + 1) % 3, i);

        if (v1 > v2)
          std::swap(v1, v2);

        // Skip if we've already checked this edge
        if (checked_edges.find({v1, v2}) != checked_edges.end()) {
          continue;
        }
        checked_edges.insert({v1, v2});

        // Calculate expected midpoint
        Vec3 expected_midpoint =
            (vertices.col(v1) + vertices.col(v2)).normalized();

        // Find the actual midpoint vertex
        Vec3 actual_midpoint;
        bool midpoint_found = false;
        for (int k = 12; k < vertices.cols(); ++k) {
          if ((vertices.col(k) - expected_midpoint).norm() < tolerance) {
            actual_midpoint = vertices.col(k);
            midpoint_found = true;
            break;
          }
        }

        INFO("Checking edge (" << v1 << ", " << v2 << ")");
        INFO("Expected midpoint: " << expected_midpoint.transpose());
        INFO("Vertex " << v1 << ": " << vertices.col(v1).transpose());
        INFO("Vertex " << v2 << ": " << vertices.col(v2).transpose());
        INFO("Vertices\n" << vertices.transpose());

        REQUIRE(midpoint_found);

        if (midpoint_found) {
          INFO("Actual midpoint: " << actual_midpoint.transpose());
          REQUIRE_THAT(actual_midpoint.norm(), WithinAbs(1.0, tolerance));
        }
      }
    }
  }
}
