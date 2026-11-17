#pragma once
#include <memory>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::opt {

/// One primitive internal coordinate. Positions are in Bohr throughout;
/// distances, translations and Cartesians are in Bohr, angles in radians,
/// and rotations in radians scaled by the fragment's radius of gyration.
struct Primitive {
  enum class Kind {
    Distance,    ///< |x_a - x_b|
    Angle,       ///< a-b-c, below ~175 degrees
    LinearBend,  ///< a-b-c near linear: e . (u_ba + u_bc) for a fixed e
    Dihedral,    ///< a-b-c-d in (-pi, pi]
    Cartesian,   ///< x_a[component]
    Translation, ///< centroid of the fragment [component]
    Rotation     ///< rotation vector of the fragment from its reference
  };
  Kind kind;
  std::vector<int> atoms; ///< the atoms involved (the fragment for
                          ///< Translation and Rotation)
  int component{0};       ///< Cartesian, Translation, Rotation: 0, 1, 2
  Vec3 direction{Vec3::Zero()}; ///< LinearBend: e; linear Rotation: axis e
  Mat3N reference;        ///< Rotation: centred reference positions
  double scale{1.0};      ///< Rotation: radius of gyration of the reference
  bool linear_fragment{false}; ///< Rotation of a linear fragment

  double value(const Mat3N &x) const;
  /// Adds d value / d x into `row` (length 3N, atom-major xyz).
  void add_derivative(const Mat3N &x, Eigen::Ref<Vec> row) const;
};

/// A set of coordinates q(x) for N atoms, with x in Bohr (3 x N).
class CoordinateSystem {
public:
  virtual ~CoordinateSystem() = default;
  virtual int size() const = 0;
  virtual Vec values(const Mat3N &x) const = 0;
  /// Wilson B matrix dq/dx, size() x 3N.
  virtual Mat wilson_b(const Mat3N &x) const = 0;
  /// q1 - q0 with periodic coordinates (dihedrals) wrapped into (-pi, pi].
  virtual Vec difference(const Vec &q1, const Vec &q0) const {
    return q1 - q0;
  }
  /// True when the coordinates describe x badly and should be rebuilt there
  /// (a bend gone linear, bonds formed or broken, a large rotation).
  virtual bool degenerate(const Mat3N &) const { return false; }
  /// Diagonal added to the model Hessian (Hartree per unit^2), for
  /// coordinates the model gives no stiffness (rigid-body motions).
  virtual Vec hessian_floor() const { return Vec::Zero(size()); }
};

class CartesianCoordinates : public CoordinateSystem {
public:
  explicit CartesianCoordinates(int num_atoms) : m_num_atoms(num_atoms) {}
  int size() const override { return 3 * m_num_atoms; }
  Vec values(const Mat3N &x) const override { return x.reshaped(); }
  Mat wilson_b(const Mat3N &) const override {
    return Mat::Identity(size(), size());
  }

private:
  int m_num_atoms;
};

/// Translation-rotation internal coordinates (TRIC; L.-P. Wang and C. Song,
/// J. Chem. Phys. 144, 214108 (2016)): distances, angles, linear bends and
/// dihedrals over the covalent bonds of each fragment, an out-of-plane
/// dihedral at every atom with three bonded neighbours, and the translation
/// and rotation of every fragment, so fragments need no artificial bonds.
/// Cartesian coordinates are added for any atom the rest leave undetermined,
/// so B has rank 3N; the coordinates count as degenerate (to be rebuilt)
/// where that rank drops.
class TRICCoordinates : public CoordinateSystem {
public:
  TRICCoordinates(const IVec &atomic_numbers, const Mat3N &x);

  int size() const override { return m_primitives.size(); }
  Vec values(const Mat3N &x) const override;
  Mat wilson_b(const Mat3N &x) const override;
  Vec difference(const Vec &q1, const Vec &q0) const override;
  bool degenerate(const Mat3N &x) const override;
  Vec hessian_floor() const override;

  const std::vector<Primitive> &primitives() const { return m_primitives; }
  int num_fragments() const { return m_num_fragments; }

private:
  IVec m_atomic_numbers;
  std::vector<std::pair<int, int>> m_bonds;
  std::vector<Primitive> m_primitives;
  int m_num_fragments{0};
  int m_rank{0};
};

/// Covalent bonds: pairs closer than 1.3 times the sum of covalent radii.
std::vector<std::pair<int, int>> covalent_bonds(const IVec &atomic_numbers,
                                                const Mat3N &x);

} // namespace occ::opt
