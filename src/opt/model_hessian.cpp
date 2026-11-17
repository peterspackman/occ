#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <occ/core/units.h>
#include <occ/opt/coordinates.h>
#include <occ/opt/model_hessian.h>
#include <vector>

namespace occ::opt {

namespace {

// Lindh's parameters by row (H-He, Li-Ne, Na and beyond), Bohr
constexpr double lindh_alpha[3][3] = {{1.0000, 0.3949, 0.3949},
                                      {0.3949, 0.2800, 0.2800},
                                      {0.3949, 0.2800, 0.2800}};
constexpr double lindh_rref[3][3] = {
    {1.35, 2.10, 2.53}, {2.10, 2.87, 3.40}, {2.53, 3.40, 3.40}};
constexpr double k_r = 0.45;
constexpr double k_f = 0.15;
constexpr double k_t = 0.005;

int row(int z) { return z <= 2 ? 0 : (z <= 10 ? 1 : 2); }

// Bends with |cos| above 0.95 (within 18 degrees of linear) take linear-bend
// terms, as in TRIC, whose linear bends would otherwise get no stiffness
// out of the plane of a nearly linear angle.
constexpr double min_sin = 0.31224989991991992; // sqrt(1 - 0.95^2)

// Add k q' q'^T for a coordinate over `atoms` with derivative columns `d`
// (Cartesian, per Bohr).
template <size_t N>
void add_term(Mat &H, const std::array<int, N> &atoms, const Mat3N &d,
              double k) {
  for (size_t a = 0; a < N; a++)
    for (size_t b = 0; b < N; b++)
      H.block<3, 3>(3 * atoms[a], 3 * atoms[b]) +=
          k * d.col(a) * d.col(b).transpose();
}

// Derivative columns (per atom of the coordinate) of a primitive.
Mat3N derivative(Primitive::Kind kind, std::vector<int> atoms,
                 const Mat3N &x) {
  Vec row = Vec::Zero(x.size());
  Primitive p{kind, atoms};
  p.add_derivative(x, row);
  Mat3N d(3, atoms.size());
  for (size_t a = 0; a < atoms.size(); a++)
    d.col(a) = row.segment<3>(3 * atoms[a]);
  return d;
}

// Rotatable bonds: covalent bonds in no ring, i.e. whose removal disconnects
// their atoms.
Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>
rotatable_bonds(const IVec &atomic_numbers, const Mat3N &bohr) {
  const int n = bohr.cols();
  const auto bonds = covalent_bonds(atomic_numbers, bohr);
  std::vector<std::vector<int>> bonded(n);
  for (const auto &[a, b] : bonds) {
    bonded[a].push_back(b);
    bonded[b].push_back(a);
  }
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> rotatable =
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(n, n,
                                                                    false);
  for (const auto &[a, b] : bonds) {
    std::vector<bool> seen(n, false);
    std::vector<int> stack{a};
    seen[a] = true;
    bool ring = false;
    while (!stack.empty() && !ring) {
      const int c = stack.back();
      stack.pop_back();
      for (int d : bonded[c]) {
        if ((c == a && d == b) || seen[d])
          continue;
        ring = ring || d == b;
        seen[d] = true;
        stack.push_back(d);
      }
    }
    rotatable(a, b) = rotatable(b, a) = !ring;
  }
  return rotatable;
}

} // namespace

Mat lindh_model_hessian(const Mat3N &positions, const IVec &atomic_numbers,
                        double threshold) {
  const int n = positions.cols();
  const Mat3N bohr = positions * occ::units::ANGSTROM_TO_BOHR;
  Mat rho = Mat::Zero(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      const int ri = row(atomic_numbers(i)), rj = row(atomic_numbers(j));
      const double r2 = (bohr.col(i) - bohr.col(j)).squaredNorm();
      const double rr = lindh_rref[ri][rj];
      rho(i, j) = rho(j, i) = std::exp(lindh_alpha[ri][rj] * (rr * rr - r2));
    }
  }

  // neighbours with rho large enough to take part in any term
  std::vector<std::vector<int>> neighbours(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j && rho(i, j) > threshold)
        neighbours[i].push_back(j);

  Mat H = Mat::Zero(3 * n, 3 * n);

  // stretches
  for (int i = 0; i < n; i++) {
    for (int j : neighbours[i]) {
      if (j < i)
        continue;
      const Vec3 u = (bohr.col(i) - bohr.col(j)).normalized();
      Mat3N d(3, 2);
      d.col(0) = u;
      d.col(1) = -u;
      add_term<2>(H, {i, j}, d, k_r * rho(i, j));
    }
  }

  auto sin_angle = [&](int i, int j, int k) {
    const Vec3 a = (bohr.col(i) - bohr.col(j)).normalized();
    const Vec3 b = (bohr.col(k) - bohr.col(j)).normalized();
    return a.cross(b).norm();
  };

  // bends i-j-k
  for (int j = 0; j < n; j++) {
    const auto &nb = neighbours[j];
    for (size_t a = 0; a < nb.size(); a++) {
      for (size_t b = a + 1; b < nb.size(); b++) {
        const int i = nb[a], k = nb[b];
        const double w = rho(i, j) * rho(j, k);
        if (w < threshold)
          continue;
        if (sin_angle(i, j, k) < min_sin) {
          // i and k on the same side of j: no bend
          if ((bohr.col(i) - bohr.col(j)).dot(bohr.col(k) - bohr.col(j)) > 0)
            continue;
          // (Near-)linear: two bends perpendicular to the axis, each the
          // sum of the end atoms' transverse displacements over their
          // distances from the centre
          const Vec3 rij = bohr.col(i) - bohr.col(j);
          const Vec3 rkj = bohr.col(k) - bohr.col(j);
          const Vec3 axis = (rij.normalized() - rkj.normalized()).normalized();
          Vec3 e1 = axis.unitOrthogonal();
          Vec3 e2 = axis.cross(e1);
          for (const Vec3 &e : {e1, e2}) {
            Mat3N d(3, 3);
            d.col(0) = e / rij.norm();
            d.col(2) = e / rkj.norm();
            d.col(1) = -(d.col(0) + d.col(2));
            add_term<3>(H, {i, j, k}, d, k_f * w);
          }
          continue;
        }
        add_term<3>(H, {i, j, k},
                    derivative(Primitive::Kind::Angle, {i, j, k}, bohr),
                    k_f * w);
      }
    }
  }

  // torsions i-j-k-l about j-k. About a rotatable bond, the terms share one
  // barrier, so their constants are divided by their number: summed, the
  // terms for every pair of substituents made such rotors 10-70 times
  // stiffer than they are (the softest modes of glycine and aspirin, against
  // GFN2-xTB Hessians), and the optimizer crept along them. Ring torsions,
  // held by the ring, keep the sum.
  const auto rotatable = rotatable_bonds(atomic_numbers, bohr);
  for (int j = 0; j < n; j++) {
    for (int k : neighbours[j]) {
      if (k < j)
        continue;
      std::vector<std::pair<std::array<int, 4>, double>> terms;
      for (int i : neighbours[j]) {
        if (i == k)
          continue;
        for (int l : neighbours[k]) {
          if (l == j || l == i)
            continue;
          const double w = rho(i, j) * rho(j, k) * rho(k, l);
          if (w < threshold || sin_angle(i, j, k) < min_sin ||
              sin_angle(j, k, l) < min_sin)
            continue;
          terms.push_back({{i, j, k, l}, w});
        }
      }
      const double scale =
          rotatable(j, k) && !terms.empty() ? 1.0 / terms.size() : 1.0;
      for (const auto &[atoms, w] : terms)
        add_term<4>(H, atoms,
                    derivative(Primitive::Kind::Dihedral,
                               {atoms.begin(), atoms.end()}, bohr),
                    k_t * w * scale);
    }
  }
  // Project out rigid translations and rotations, which the near-linear
  // bends do not quite avoid: H <- P H P with P = 1 - T T^T.
  const Vec3 com = bohr.rowwise().mean();
  Mat T = Mat::Zero(3 * n, 6);
  for (int a = 0; a < 3; a++) {
    Vec3 axis = Vec3::Zero();
    axis(a) = 1.0;
    for (int i = 0; i < n; i++) {
      T(3 * i + a, a) = 1.0;
      T.block<3, 1>(3 * i, 3 + a) = axis.cross(Vec3(bohr.col(i) - com));
    }
  }
  Mat Q(3 * n, 0);
  for (int c = 0; c < 6; c++) {
    Vec v = T.col(c);
    for (int q = 0; q < Q.cols(); q++)
      v -= Q.col(q).dot(v) * Q.col(q);
    if (v.norm() > 1e-6 * T.col(c).norm() && v.norm() > 1e-10) {
      Q.conservativeResize(Eigen::NoChange, Q.cols() + 1);
      Q.col(Q.cols() - 1) = v.normalized();
    }
  }
  const Mat HQ = H * Q;
  H -= HQ * Q.transpose() + Q * HQ.transpose() -
       Q * (Q.transpose() * HQ) * Q.transpose();
  return H;
}

} // namespace occ::opt
