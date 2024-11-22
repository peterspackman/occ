#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

/**
 * Class representing and holding data for an Atom in 3D space.
 *
 * Small, minimal structure designed to hold very little information
 * and be trivially copyable.
 */
struct Atom {
  /// the atomic number of this atom i.e. which element it is
  int atomic_number;
  /// 3D coordinates in Bohr
  double x, y, z;

  /// rotate the position of this atom about the origin by the provided
  /// rotation matrix
  inline void rotate(const Mat3 &rotation) {
    occ::Vec3 pos{x, y, z};
    auto pos_rot = rotation * pos;
    x = pos_rot(0);
    y = pos_rot(1);
    z = pos_rot(2);
  }

  /// translate the position of this atom by the provided translation (Bohr)
  inline void translate(const Vec3 &translation) {
    x += translation(0);
    y += translation(1);
    z += translation(2);
  }

  /// convenience helper to convert this position into \a Vec3
  inline Vec3 position() const { return {x, y, z}; }

  /// convenience helper to set this position from \a Vec3
  inline void set_position(const Vec3 &v) {
    x = v.x();
    y = v.y();
    z = v.z();
  }

  /// the square euclidean distance from another point in space (Bohr^2)
  inline double square_distance(double xx, double yy, double zz) const {
    double dx = xx - x, dy = yy - y, dz = zz - z;
    return dx * dx + dy * dy + dz * dz;
  }

  /// the square euclidean distance from another atom (Bohr^2)
  inline double square_distance(const Atom &other) const {
    double dx = other.x - x, dy = other.y - y, dz = other.z - z;
    return dx * dx + dy * dy + dz * dz;
  }
};

/// returns true if two Atoms have the same element and are at the exact same
/// point in space
inline bool operator==(const Atom &atom1, const Atom &atom2) {
  return atom1.atomic_number == atom2.atomic_number && atom1.x == atom2.x &&
         atom1.y == atom2.y && atom1.z == atom2.z;
}

/// Rotate a range of atoms about the origin
template <typename AtomIterator>
inline void rotate_atoms(AtomIterator &atoms, const Mat3 &rotation) {
  for (auto &atom : atoms) {
    atom.rotate(rotation);
  }
}

/// translate a range of atoms
template <typename AtomIterator>
inline void translate_atoms(AtomIterator &atoms, const Vec3 &translation) {
  for (auto &atom : atoms) {
    atom.translate(translation);
  }
}

} // namespace occ::core
