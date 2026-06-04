#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <occ/geometry/half_edge.h>
#include <occ/isosurface/improve_quality.h>
#include <occ/isosurface/refine_sharp.h>

using occ::FVec3;
using Catch::Approx;

namespace {
// Implicit sphere of radius R: zero set is |p| = R (not an SDF, so projection
// takes a couple of Newton steps).
struct SphereField {
  float R;
  float operator()(const FVec3 &p) const { return p.squaredNorm() - R * R; }
};

int count_boundary_edges(const std::vector<uint32_t> &F, int &nonmanifold) {
  std::map<std::pair<uint32_t, uint32_t>, int> ec;
  for (size_t f = 0; f < F.size() / 3; f++)
    for (int e = 0; e < 3; e++) {
      uint32_t a = F[3 * f + e], b = F[3 * f + (e + 1) % 3];
      ec[{std::min(a, b), std::max(a, b)}]++;
    }
  int bnd = 0;
  nonmanifold = 0;
  for (auto &[k, c] : ec) {
    if (c == 1)
      bnd++;
    if (c > 2)
      nonmanifold++;
  }
  return bnd;
}

// A closed octahedron of radius R inscribed on the sphere.
void make_octahedron(float R, std::vector<float> &V, std::vector<float> &N,
                     std::vector<float> &C, std::vector<uint32_t> &F) {
  V = {R, 0, 0, -R, 0, 0, 0, R, 0, 0, -R, 0, 0, 0, R, 0, 0, -R};
  N.clear();
  C.clear();
  for (int i = 0; i < 6; i++) {
    FVec3 p(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    FVec3 n = p.normalized();
    N.insert(N.end(), {n[0], n[1], n[2]});
    C.insert(C.end(), {0.0f, 0.0f});
  }
  F = {4, 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0,
       5, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3};
}
} // namespace

TEST_CASE("Sharp refinement projects onto the surface and stays manifold",
          "[isosurface][refine]") {
  // Octahedron: adjacent faces meet at a large dihedral, so a low angle
  // threshold flags every edge.
  const float R = 2.0f;
  std::vector<float> V, N, C;
  std::vector<uint32_t> F;
  make_octahedron(R, V, N, C, F);

  SphereField field{R};
  occ::isosurface::SharpRefineParams params;
  params.passes = 1;
  params.angle_threshold_deg = 1.0f; // flag every edge
  params.projection_steps = 5;

  occ::isosurface::refine_sharp_edges(field, 0.0f, params, V, N, C, F);

  // 8 faces, all 12 edges split -> classic 1->4 -> 32 faces, 6+12 = 18 verts.
  REQUIRE(F.size() / 3 == 32);
  REQUIRE(V.size() / 3 == 18);

  for (size_t i = 0; i < V.size() / 3; i++) {
    FVec3 p(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    REQUIRE(p.norm() == Approx(R).margin(1e-4));
  }

  int nonmanifold = 0;
  REQUIRE(count_boundary_edges(F, nonmanifold) == 0);
  REQUIRE(nonmanifold == 0);
}

TEST_CASE("Half-edge flip is reversible and preserves orientation",
          "[geometry][halfedge]") {
  using occ::geometry::HalfEdgeMesh;
  // Two triangles forming a unit square; only the diagonal 0-2 is interior.
  std::vector<float> Vpos = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  std::vector<uint32_t> F = {0, 1, 2, 0, 2, 3};

  HalfEdgeMesh mesh(4, F);
  REQUIRE(mesh.check_valid());

  int e = -1;
  for (int h = 0; h < mesh.n_halfedges(); h++)
    if (!mesh.is_boundary(h)) {
      e = h;
      break;
    }
  REQUIRE(e >= 0);

  REQUIRE(mesh.flip(e));
  REQUIRE(mesh.check_valid());
  REQUIRE(mesh.edge_exists(1, 3));
  REQUIRE_FALSE(mesh.edge_exists(0, 2));

  // Orientation preserved: both faces still face +z.
  auto vat = [&](int i) { return FVec3(Vpos[3 * i], Vpos[3 * i + 1], 0.0f); };
  std::vector<uint32_t> Fout;
  mesh.export_faces(Fout);
  for (size_t f = 0; f < Fout.size() / 3; f++) {
    FVec3 n = (vat(Fout[3 * f + 1]) - vat(Fout[3 * f]))
                  .cross(vat(Fout[3 * f + 2]) - vat(Fout[3 * f]));
    REQUIRE(n.z() > 0.0f);
  }

  // Flipping again restores the original face set.
  REQUIRE(mesh.flip(e));
  REQUIRE(mesh.edge_exists(0, 2));
  REQUIRE_FALSE(mesh.edge_exists(1, 3));
}

TEST_CASE("Quality flips raise the minimum triangle angle",
          "[isosurface][quality]") {
  // A thin quad with apexes on opposite sides of the diagonal (consistent
  // winding): the diagonal 0-2 yields two slivers; flipping to 1-3 gives two
  // fat triangles.
  std::vector<float> V = {0, 0, 0, 1, -0.05f, 0, 2, 0, 0, 1, 1.0f, 0};
  std::vector<float> N = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
  std::vector<float> C = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint32_t> F = {0, 1, 2, 0, 2, 3};

  float before = occ::isosurface::triangle_angle_stats(V, F).min_deg;

  occ::isosurface::QualityParams params;
  params.iterations = 3;
  params.feature_angle_deg = 180.0f; // allow any flip in this flat test
  SphereField dummy{1.0f};           // relaxation off -> field is unused
  occ::isosurface::improve_mesh_quality(dummy, 0.0f, params, V, N, C, F);

  float after = occ::isosurface::triangle_angle_stats(V, F).min_deg;
  REQUIRE(after > before);
  REQUIRE(F.size() / 3 == 2); // flips never change the face count
}

TEST_CASE("Tangential relaxation keeps vertices on the surface",
          "[isosurface][quality]") {
  const float R = 2.0f;
  std::vector<float> V, N, C;
  std::vector<uint32_t> F;
  make_octahedron(R, V, N, C, F);
  SphereField field{R};

  // Subdivide to a denser (non-uniform) sphere mesh first.
  occ::isosurface::SharpRefineParams sp;
  sp.passes = 2;
  sp.angle_threshold_deg = 1.0f;
  sp.projection_steps = 5;
  occ::isosurface::refine_sharp_edges(field, 0.0f, sp, V, N, C, F);

  occ::isosurface::QualityParams qp;
  qp.iterations = 5;
  qp.feature_angle_deg = 180.0f; // smooth sphere: nothing is a feature
  qp.relaxation_factor = 0.5f;
  qp.projection_steps = 4;
  occ::isosurface::improve_mesh_quality(field, 0.0f, qp, V, N, C, F);

  for (size_t i = 0; i < V.size() / 3; i++) {
    FVec3 p(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    REQUIRE(p.norm() == Approx(R).margin(1e-2));
  }
}

TEST_CASE("Short-edge collapse removes vertices and stays manifold",
          "[isosurface][quality]") {
  const float R = 2.0f;
  std::vector<float> V, N, C;
  std::vector<uint32_t> F;
  make_octahedron(R, V, N, C, F);
  SphereField field{R};

  const size_t before = V.size() / 3;
  // ratio 5 -> every edge is "short", so collapses fire (subject to the link
  // condition / inversion guards).
  int n =
      occ::isosurface::impl::collapse_short_edges(field, 0.0f, V, N, C, F, 5.0f);
  REQUIRE(n > 0);
  REQUIRE(V.size() / 3 < before);

  // Still closed and manifold (every edge shared by exactly two faces), no
  // degenerate faces, survivors on the sphere.
  std::map<std::pair<uint32_t, uint32_t>, int> ec;
  for (size_t f = 0; f < F.size() / 3; f++) {
    REQUIRE(F[3 * f] != F[3 * f + 1]);
    REQUIRE(F[3 * f + 1] != F[3 * f + 2]);
    REQUIRE(F[3 * f] != F[3 * f + 2]);
    for (int e = 0; e < 3; e++) {
      uint32_t a = F[3 * f + e], b = F[3 * f + (e + 1) % 3];
      ec[{std::min(a, b), std::max(a, b)}]++;
    }
  }
  for (auto &[k, c] : ec)
    REQUIRE(c == 2);
  for (size_t i = 0; i < V.size() / 3; i++) {
    FVec3 p(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
    REQUIRE(p.norm() == Approx(R).margin(1e-2));
  }
}
