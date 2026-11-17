#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <catch2/catch_all.hpp>
#include <occ/core/units.h>
#include <occ/opt/coordinates.h>

using occ::IVec;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;
using occ::opt::Primitive;
using occ::opt::TRICCoordinates;

namespace {

struct System {
  IVec z;
  Mat3N x; // Bohr
};

System make(std::initializer_list<int> z,
            std::initializer_list<std::array<double, 3>> angstrom) {
  System s;
  s.z = IVec(z.size());
  int i = 0;
  for (int zi : z)
    s.z(i++) = zi;
  s.x = Mat3N(3, angstrom.size());
  i = 0;
  for (const auto &p : angstrom)
    s.x.col(i++) = Vec3(p[0], p[1], p[2]) * occ::units::ANGSTROM_TO_BOHR;
  return s;
}

System ethanol() {
  return make({6, 6, 8, 1, 1, 1, 1, 1, 1},
              {{-1.168, -0.408, 0.000},
               {0.255, 0.161, 0.000},
               {0.951, -0.890, 0.000},
               {-1.952, 0.315, 0.000},
               {-1.271, -0.968, 0.893},
               {-1.235, 0.212, -0.884},
               {0.320, 0.787, 0.891},
               {0.312, 0.844, -0.876},
               {1.829, -0.469, 0.004}});
}

System acetylene() {
  return make({1, 6, 6, 1},
              {{0, 0, -1.66}, {0, 0, -0.60}, {0, 0.01, 0.60}, {0, 0, 1.66}});
}

System water_trimer() {
  return make({8, 1, 1, 8, 1, 1, 8, 1, 1},
              {{0.000, 0.000, 0.000},
               {0.757, 0.586, 0.000},
               {-0.757, 0.586, 0.000},
               {3.600, 0.000, 0.000},
               {3.200, 0.700, 0.300},
               {4.300, 0.100, 0.600},
               {1.800, 3.200, 0.500},
               {1.300, 2.500, 0.100},
               {2.500, 3.300, -0.200}});
}

System butyne() {
  // CH3-C#C-CH3: the torsion of the methyls about the linear chain
  return make({6, 6, 6, 6, 1, 1, 1, 1, 1, 1},
              {{0, 0, -2.07},
               {0, 0, -0.60},
               {0, 0, 0.60},
               {0, 0, 2.07},
               {1.02, 0, -2.46},
               {-0.51, 0.88, -2.46},
               {-0.51, -0.88, -2.46},
               {1.02, 0.1, 2.46},
               {-0.43, 0.93, 2.46},
               {-0.59, -0.83, 2.46}});
}

// Rank of B by the relative cutoff the optimizer uses.
int rank_of(const Mat &B) {
  Eigen::SelfAdjointEigenSolver<Mat> es(B.transpose() * B);
  const Vec &ev = es.eigenvalues();
  int rank = 0;
  for (int i = 0; i < ev.size(); i++)
    rank += ev(i) > 1e-8 * ev(ev.size() - 1);
  return rank;
}

// A slightly distorted copy, so no coordinate sits at a special value.
Mat3N jiggle(Mat3N x, double amount = 0.05) {
  for (int i = 0; i < x.cols(); i++)
    x.col(i) += amount * Vec3(std::sin(1.3 * i + 0.1), std::cos(2.1 * i),
                              std::sin(0.7 * i + 1.0));
  return x;
}

} // namespace

TEST_CASE("TRIC Wilson B matches finite differences", "[tric]") {
  for (const auto &s : {ethanol(), acetylene(), water_trimer(), butyne()}) {
    TRICCoordinates coords(s.z, s.x);
    const Mat3N x = jiggle(s.x, 0.02);
    const Mat B = coords.wilson_b(x);
    constexpr double h = 1e-5;
    Mat fd(B.rows(), B.cols());
    for (int j = 0; j < x.size(); j++) {
      Mat3N xp = x, xm = x;
      xp(j % 3, j / 3) += h;
      xm(j % 3, j / 3) -= h;
      fd.col(j) = coords.difference(coords.values(xp), coords.values(xm)) /
                  (2 * h);
    }
    for (int i = 0; i < B.rows(); i++) {
      CAPTURE(i, static_cast<int>(coords.primitives()[i].kind));
      REQUIRE((B.row(i) - fd.row(i)).norm() <
              1e-6 * std::max(1.0, B.row(i).norm()));
    }
  }
}

TEST_CASE("TRIC coordinates have rank 3N", "[tric]") {
  const auto he = make({2}, {{0, 0, 0}});
  const auto h2 = make({1, 1}, {{0, 0, 0}, {0, 0, 0.74}});
  for (const auto &s :
       {ethanol(), acetylene(), water_trimer(), butyne(), he, h2}) {
    TRICCoordinates coords(s.z, s.x);
    CAPTURE(s.x.cols(), coords.size());
    REQUIRE(rank_of(coords.wilson_b(s.x)) == s.x.size());
  }
}

TEST_CASE("TRIC fragments are found from covalent bonds", "[tric]") {
  TRICCoordinates coords(water_trimer().z, water_trimer().x);
  REQUIRE(coords.num_fragments() == 3);
  int distances = 0;
  for (const auto &p : coords.primitives())
    distances += p.kind == Primitive::Kind::Distance;
  REQUIRE(distances == 6); // only the O-H bonds, no intermolecular bonds
}

TEST_CASE("TRIC rotation coordinates recover a rigid rotation",
          "[tric]") {
  const auto s = ethanol();
  TRICCoordinates coords(s.z, s.x);
  const Vec3 v(0.2, -0.3, 0.1);
  const Eigen::AngleAxisd R(v.norm(), v.normalized());
  const Vec3 c = s.x.rowwise().mean();
  const Mat3N x = (R.toRotationMatrix() * (s.x.colwise() - c)).colwise() + c;
  const Vec q = coords.values(x);
  Vec3 found = Vec3::Zero();
  for (size_t i = 0; i < coords.primitives().size(); i++) {
    const auto &p = coords.primitives()[i];
    if (p.kind == Primitive::Kind::Rotation)
      found(p.component) = q(i) / p.scale;
    if (p.kind == Primitive::Kind::Distance || p.kind == Primitive::Kind::Angle)
      REQUIRE(q(i) == Catch::Approx(p.value(s.x)).margin(1e-10));
  }
  REQUIRE((found - v).norm() < 1e-10);
}

TEST_CASE("TRIC dihedral differences wrap", "[tric]") {
  const auto s = ethanol();
  TRICCoordinates coords(s.z, s.x);
  Vec q0 = Vec::Zero(coords.size()), q1 = Vec::Zero(coords.size());
  for (size_t i = 0; i < coords.primitives().size(); i++) {
    if (coords.primitives()[i].kind == Primitive::Kind::Dihedral) {
      q0(i) = 3.1;
      q1(i) = -3.1;
    }
  }
  const Vec d = coords.difference(q1, q0);
  for (size_t i = 0; i < coords.primitives().size(); i++)
    if (coords.primitives()[i].kind == Primitive::Kind::Dihedral)
      REQUIRE(d(i) == Catch::Approx(2 * M_PI - 6.2).margin(1e-12));
}

TEST_CASE("TRIC asks for a rebuild when a bend goes linear", "[tric]") {
  const auto s = make({8, 1, 1}, {{0, 0, 0}, {0.96, 0, 0}, {-0.24, 0.93, 0}});
  TRICCoordinates coords(s.z, s.x);
  REQUIRE_FALSE(coords.degenerate(s.x));
  const Mat3N linear = make({8, 1, 1}, {{0, 0, 0}, {0.96, 0, 0}, {-0.96, 0.01, 0}}).x;
  REQUIRE(coords.degenerate(linear));
}
