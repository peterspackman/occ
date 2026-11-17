#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/opt/coordinates.h>

namespace occ::opt {

namespace {

constexpr double pi = std::numbers::pi_v<double>;
// Bends above this (|cos| > 0.95, as geomeTRIC) are described by linear
// bends rather than an angle, and do not carry dihedrals.
const double linear_angle = std::acos(-0.95);
// Rebuild when an angle (or a dihedral's bend) passes rebuild_angle, where
// its derivative grows large, or a linear bend closes below bent_angle.
constexpr double rebuild_angle = 175.0 * pi / 180.0;
constexpr double bent_angle = 150.0 * pi / 180.0;
// Rebuild when a linear bend's direction has turned this far towards the
// axis (cosine), as the molecule rotates.
constexpr double max_bend_tilt = 0.3;
// Rebuild when a fragment has turned this far (radians, per component) from
// its reference, well before the rotation vector's branch at pi.
constexpr double max_rotation = 1.0;
// Fragments whose atoms lie this close to a line (sqrt of the ratio of the
// two larger second moments) rotate about two axes only.
constexpr double linear_fragment = 0.05;
// Least model-Hessian stiffness, so no coordinate starts with none: for
// rigid-body coordinates, which the model leaves free between molecules out
// of contact, about a hydrogen bond's (geomeTRIC's guess of 0.05 is the
// stiffness of a double hydrogen bond, and made the water dimer and the
// dispersion-bound complexes take up to twice the steps); and for bends and
// torsions.
constexpr double rigid_body_floor = 0.01;
constexpr double internal_floor = 0.005;

double angle(const Vec3 &u, const Vec3 &v) {
  return std::atan2(u.cross(v).norm(), u.dot(v));
}

double bend(const Mat3N &x, int a, int b, int c) {
  return angle(x.col(a) - x.col(b), x.col(c) - x.col(b));
}

Mat3N centred(const Mat3N &x, const std::vector<int> &atoms) {
  Mat3N p(3, atoms.size());
  for (size_t i = 0; i < atoms.size(); i++)
    p.col(i) = x.col(atoms[i]);
  return p.colwise() - p.rowwise().mean();
}

// Principal axis (largest second moment) of centred positions.
Vec3 principal_axis(const Mat3N &p) {
  Eigen::SelfAdjointEigenSolver<Mat3> es(p * p.transpose());
  return es.eigenvectors().col(2);
}

// Rotation vector of the rotation R minimising sum |R p0_i - p_i|^2 (Horn's
// quaternion method), for centred positions.
Vec3 rotation_vector(const Mat3N &p0, const Mat3N &p) {
  const Mat3 S = p0 * p.transpose();
  Eigen::Matrix4d N;
  N << S(0, 0) + S(1, 1) + S(2, 2), S(1, 2) - S(2, 1), S(2, 0) - S(0, 2),
      S(0, 1) - S(1, 0), //
      S(1, 2) - S(2, 1), S(0, 0) - S(1, 1) - S(2, 2), S(0, 1) + S(1, 0),
      S(2, 0) + S(0, 2), //
      S(2, 0) - S(0, 2), S(0, 1) + S(1, 0), -S(0, 0) + S(1, 1) - S(2, 2),
      S(1, 2) + S(2, 1), //
      S(0, 1) - S(1, 0), S(2, 0) + S(0, 2), S(1, 2) + S(2, 1),
      -S(0, 0) - S(1, 1) + S(2, 2);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(N);
  Eigen::Vector4d q = es.eigenvectors().col(3);
  if (q(0) < 0)
    q = -q;
  const Vec3 axis = q.tail<3>();
  const double s = axis.norm();
  if (s < 1e-14)
    return 2.0 * axis;
  return 2.0 * std::atan2(s, q(0)) * axis / s;
}

// Rotation taking axis a0 to a (both unit), as a rotation vector.
Vec3 axis_rotation(const Vec3 &a0, Vec3 a) {
  if (a.dot(a0) < 0)
    a = -a;
  const Vec3 n = a0.cross(a);
  const double s = n.norm();
  if (s < 1e-14)
    return n;
  return std::atan2(s, a0.dot(a)) * n / s;
}

double rotation_value(const Primitive &p, const Mat3N &x) {
  const Mat3N current = centred(x, p.atoms);
  if (p.linear_fragment) {
    const Vec3 a0 = principal_axis(p.reference);
    const Vec3 v = axis_rotation(a0, principal_axis(current));
    const Vec3 e = p.component == 0 ? p.direction
                                    : Vec3(a0.cross(p.direction));
    return p.scale * v.dot(e);
  }
  return p.scale * rotation_vector(p.reference, current)(p.component);
}

} // namespace

double Primitive::value(const Mat3N &x) const {
  switch (kind) {
  case Kind::Distance:
    return (x.col(atoms[0]) - x.col(atoms[1])).norm();
  case Kind::Angle:
    return bend(x, atoms[0], atoms[1], atoms[2]);
  case Kind::LinearBend: {
    const Vec3 u = (x.col(atoms[0]) - x.col(atoms[1])).normalized();
    const Vec3 w = (x.col(atoms[2]) - x.col(atoms[1])).normalized();
    return direction.dot(u + w);
  }
  case Kind::Dihedral: {
    const Vec3 F = x.col(atoms[0]) - x.col(atoms[1]);
    const Vec3 G = x.col(atoms[1]) - x.col(atoms[2]);
    const Vec3 H = x.col(atoms[3]) - x.col(atoms[2]);
    const Vec3 A = F.cross(G), B = H.cross(G);
    return std::atan2(B.cross(A).dot(G) / G.norm(), A.dot(B));
  }
  case Kind::Cartesian:
    return x(component, atoms[0]);
  case Kind::Translation: {
    double sum = 0.0;
    for (int a : atoms)
      sum += x(component, a);
    return sum / atoms.size();
  }
  case Kind::Rotation:
    return rotation_value(*this, x);
  }
  return 0.0;
}

void Primitive::add_derivative(const Mat3N &x, Eigen::Ref<Vec> row) const {
  auto add = [&](int atom, const Vec3 &d) { row.segment<3>(3 * atom) += d; };
  switch (kind) {
  case Kind::Distance: {
    const Vec3 u = (x.col(atoms[0]) - x.col(atoms[1])).normalized();
    add(atoms[0], u);
    add(atoms[1], -u);
    break;
  }
  case Kind::Angle: {
    const Vec3 u = x.col(atoms[0]) - x.col(atoms[1]);
    const Vec3 v = x.col(atoms[2]) - x.col(atoms[1]);
    const double theta = angle(u, v);
    const double c = std::cos(theta), s = std::sin(theta);
    const Vec3 uh = u.normalized(), vh = v.normalized();
    const Vec3 da = (c * uh - vh) / (u.norm() * s);
    const Vec3 dc = (c * vh - uh) / (v.norm() * s);
    add(atoms[0], da);
    add(atoms[2], dc);
    add(atoms[1], -(da + dc));
    break;
  }
  case Kind::LinearBend: {
    const Vec3 u = x.col(atoms[0]) - x.col(atoms[1]);
    const Vec3 w = x.col(atoms[2]) - x.col(atoms[1]);
    const Vec3 uh = u.normalized(), wh = w.normalized();
    const Vec3 da = (direction - uh * uh.dot(direction)) / u.norm();
    const Vec3 dc = (direction - wh * wh.dot(direction)) / w.norm();
    add(atoms[0], da);
    add(atoms[2], dc);
    add(atoms[1], -(da + dc));
    break;
  }
  case Kind::Dihedral: {
    // Blondel and Karplus, J. Comput. Chem. 17, 1132 (1996)
    const Vec3 F = x.col(atoms[0]) - x.col(atoms[1]);
    const Vec3 G = x.col(atoms[1]) - x.col(atoms[2]);
    const Vec3 H = x.col(atoms[3]) - x.col(atoms[2]);
    const Vec3 A = F.cross(G), B = H.cross(G);
    const double g = G.norm(), a2 = A.squaredNorm(), b2 = B.squaredNorm();
    const Vec3 dA = -g / a2 * A; // d phi / d x_a
    const Vec3 dD = g / b2 * B;  // d phi / d x_d
    const Vec3 t = F.dot(G) / (a2 * g) * A - H.dot(G) / (b2 * g) * B;
    add(atoms[0], dA);
    add(atoms[1], -dA + t);
    add(atoms[2], -dD - t);
    add(atoms[3], dD);
    break;
  }
  case Kind::Cartesian:
    row(3 * atoms[0] + component) += 1.0;
    break;
  case Kind::Translation:
    for (int a : atoms)
      row(3 * a + component) += 1.0 / atoms.size();
    break;
  case Kind::Rotation: {
    // central differences; the alignment has no simple closed form
    constexpr double h = 1e-4;
    Mat3N xp = x;
    for (int a : atoms) {
      for (int k = 0; k < 3; k++) {
        const double x0 = xp(k, a);
        xp(k, a) = x0 + h;
        const double fp = value(xp);
        xp(k, a) = x0 - h;
        const double fm = value(xp);
        xp(k, a) = x0;
        row(3 * a + k) += (fp - fm) / (2 * h);
      }
    }
    break;
  }
  }
}

std::vector<std::pair<int, int>> covalent_bonds(const IVec &atomic_numbers,
                                                const Mat3N &x) {
  std::vector<std::pair<int, int>> bonds;
  const int n = x.cols();
  std::vector<double> radius(n);
  for (int i = 0; i < n; i++)
    radius[i] = occ::core::Element(atomic_numbers(i)).covalent_radius() *
                occ::units::ANGSTROM_TO_BOHR;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      if ((x.col(i) - x.col(j)).norm() < 1.3 * (radius[i] + radius[j]))
        bonds.emplace_back(i, j);
  return bonds;
}

TRICCoordinates::TRICCoordinates(const IVec &atomic_numbers, const Mat3N &x)
    : m_atomic_numbers(atomic_numbers),
      m_bonds(covalent_bonds(atomic_numbers, x)) {
  const int n = x.cols();
  std::vector<std::vector<int>> neighbours(n);
  for (const auto &[i, j] : m_bonds) {
    neighbours[i].push_back(j);
    neighbours[j].push_back(i);
  }

  for (const auto &[i, j] : m_bonds)
    m_primitives.push_back({Primitive::Kind::Distance, {i, j}});

  for (int b = 0; b < n; b++) {
    const auto &nb = neighbours[b];
    for (size_t i = 0; i < nb.size(); i++) {
      for (size_t j = i + 1; j < nb.size(); j++) {
        const int a = nb[i], c = nb[j];
        if (bend(x, a, b, c) < linear_angle) {
          m_primitives.push_back({Primitive::Kind::Angle, {a, b, c}});
          continue;
        }
        const Vec3 axis = (x.col(c) - x.col(a)).normalized();
        const Vec3 e1 = axis.unitOrthogonal();
        for (const Vec3 &e : {e1, Vec3(axis.cross(e1))}) {
          Primitive p{Primitive::Kind::LinearBend, {a, b, c}};
          p.direction = e;
          m_primitives.push_back(p);
        }
      }
    }
  }

  for (const auto &[b, c] : m_bonds) {
    for (int a : neighbours[b]) {
      if (a == c || bend(x, a, b, c) >= linear_angle)
        continue;
      for (int d : neighbours[c]) {
        if (d == b || d == a || bend(x, b, c, d) >= linear_angle)
          continue;
        m_primitives.push_back({Primitive::Kind::Dihedral, {a, b, c, d}});
      }
    }
  }

  // fragments: connected components of the bonds
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  };
  for (const auto &[i, j] : m_bonds)
    parent[find(i)] = find(j);
  std::vector<std::vector<int>> fragments;
  {
    std::vector<int> index(n, -1);
    for (int i = 0; i < n; i++) {
      const int r = find(i);
      if (index[r] < 0) {
        index[r] = fragments.size();
        fragments.emplace_back();
      }
      fragments[index[r]].push_back(i);
    }
  }
  m_num_fragments = fragments.size();

  for (const auto &fragment : fragments) {
    for (int k = 0; k < 3; k++) {
      Primitive p{Primitive::Kind::Translation, fragment};
      p.component = k;
      m_primitives.push_back(p);
    }
    if (fragment.size() < 2)
      continue;
    const Mat3N reference = centred(x, fragment);
    Eigen::SelfAdjointEigenSolver<Mat3> es(reference * reference.transpose());
    const Vec3 moments = es.eigenvalues().cwiseMax(0.0);
    const bool linear = std::sqrt(moments(1) / moments(2)) < linear_fragment;
    const double rg = std::sqrt(reference.colwise().squaredNorm().mean());
    for (int k = 0; k < (linear ? 2 : 3); k++) {
      Primitive p{Primitive::Kind::Rotation, fragment};
      p.component = k;
      p.reference = reference;
      p.scale = rg;
      p.linear_fragment = linear;
      if (linear)
        p.direction = principal_axis(reference).unitOrthogonal();
      m_primitives.push_back(p);
    }
  }

  // Cartesian coordinates for the atoms whose motion the rest leave
  // undetermined (e.g. torsions about linear chains), until B has rank 3N.
  for (int pass = 0; pass < n; pass++) {
    const Mat B = wilson_b(x);
    Eigen::SelfAdjointEigenSolver<Mat> es(B.transpose() * B);
    const Vec &ev = es.eigenvalues();
    int null = 0;
    while (null < ev.size() && ev(null) < 1e-8 * ev(ev.size() - 1))
      null++;
    if (null == 0)
      break;
    const Mat V = es.eigenvectors().leftCols(null);
    Vec weight = V.rowwise().squaredNorm().reshaped(3, n).colwise().sum();
    const double cutoff = std::min(0.05, weight.maxCoeff());
    int added = 0;
    for (int a = 0; a < n; a++) {
      if (weight(a) < cutoff)
        continue;
      for (int k = 0; k < 3; k++) {
        Primitive p{Primitive::Kind::Cartesian, {a}};
        p.component = k;
        m_primitives.push_back(p);
      }
      added++;
    }
    log::debug("TRIC: {} null directions, added Cartesians for {} atoms",
               null, added);
  }

  log::debug("TRIC coordinates: {} primitives over {} fragments",
             m_primitives.size(), m_num_fragments);
}

Vec TRICCoordinates::values(const Mat3N &x) const {
  Vec q(m_primitives.size());
  for (size_t i = 0; i < m_primitives.size(); i++)
    q(i) = m_primitives[i].value(x);
  return q;
}

Mat TRICCoordinates::wilson_b(const Mat3N &x) const {
  Mat B = Mat::Zero(m_primitives.size(), x.size());
  Vec row(x.size());
  for (size_t i = 0; i < m_primitives.size(); i++) {
    row.setZero();
    m_primitives[i].add_derivative(x, row);
    B.row(i) = row.transpose();
  }
  return B;
}

Vec TRICCoordinates::difference(const Vec &q1, const Vec &q0) const {
  Vec d = q1 - q0;
  for (size_t i = 0; i < m_primitives.size(); i++)
    if (m_primitives[i].kind == Primitive::Kind::Dihedral)
      d(i) = std::remainder(d(i), 2 * pi);
  return d;
}

bool TRICCoordinates::degenerate(const Mat3N &x) const {
  if (covalent_bonds(m_atomic_numbers, x) != m_bonds) {
    log::debug("TRIC: bonding changed");
    return true;
  }
  for (const auto &p : m_primitives) {
    const auto &a = p.atoms;
    switch (p.kind) {
    case Primitive::Kind::Angle:
      if (bend(x, a[0], a[1], a[2]) > rebuild_angle)
        return true;
      break;
    case Primitive::Kind::LinearBend:
      if (bend(x, a[0], a[1], a[2]) < bent_angle ||
          std::abs(p.direction.dot(
              (x.col(a[2]) - x.col(a[0])).normalized())) > max_bend_tilt)
        return true;
      break;
    case Primitive::Kind::Dihedral:
      if (bend(x, a[0], a[1], a[2]) > rebuild_angle ||
          bend(x, a[1], a[2], a[3]) > rebuild_angle)
        return true;
      break;
    case Primitive::Kind::Rotation:
      if (std::abs(p.value(x)) > max_rotation * p.scale)
        return true;
      break;
    default:
      break;
    }
  }
  return false;
}

Vec TRICCoordinates::hessian_floor() const {
  Vec floor = Vec::Zero(m_primitives.size());
  for (size_t i = 0; i < m_primitives.size(); i++) {
    switch (m_primitives[i].kind) {
    case Primitive::Kind::Translation:
    case Primitive::Kind::Rotation:
      floor(i) = rigid_body_floor;
      break;
    case Primitive::Kind::Angle:
    case Primitive::Kind::LinearBend:
    case Primitive::Kind::Dihedral:
      floor(i) = internal_floor;
      break;
    default:
      break;
    }
  }
  return floor;
}

} // namespace occ::opt
