#pragma once
#include <optional>
#include <trajan/core/element.h>
#include <trajan/core/linear_algebra.h>

namespace trajan::core {

struct Bond {
  std::pair<size_t, size_t> idxs;
  double bond_length;
  Bond() : bond_length(0.0), idxs({0, 0}) {}
  Bond(double bond_length) : bond_length(bond_length), idxs({0, 0}) {}
};

struct Atom {
  double x, y, z;
  Element element;
  std::string type;
  size_t index, serial;
  Atom() : element(0), index(0) {}
  Atom(const Vec3 &pos, int atomic_number, int index)
      : x(pos[0]), y(pos[1]), z(pos[2]),
        element(static_cast<Element>(atomic_number)), index(index) {}
  Atom(const Vec3 &pos, std::string element_type, int index)
      : x(pos[0]), y(pos[1]), z(pos[2]),
        element(static_cast<Element>(element_type)), index(index) {}
  Atom(const Vec3 &pos, Element element, int index)
      : x(pos[0]), y(pos[1]), z(pos[2]), element(element), index(index) {}
  Atom(const Atom &other, Vec3 shift)
      : x(other.x + shift.x()), y(other.y + shift.y()), z(other.z + shift.z()),
        element(other.element), index(other.index) {}

  inline Atom create_ghost(Vec3 shift) const { return Atom(*this, shift); }

  inline void update_position(const Vec3 &pos) {
    x = pos.x(), y = pos.y(), z = pos.z();
  }

  inline Vec3 position() const { return {x, y, z}; }

  // inline size_t &index() const { return index; }

  inline double square_distance(const Atom &other) const {
    double dx = other.x - x, dy = other.y - y, dz = other.z - z;
    return dx * dx + dy * dy + dz * dz;
  }

  inline void rotate(const trajan::Mat3 &rotation) {
    trajan::Vec3 pos{x, y, z};
    auto pos_rot = rotation * pos;
    x = pos_rot(0);
    y = pos_rot(1);
    z = pos_rot(2);
  }

  inline void translate(const trajan::Vec3 &translation) {
    x += translation(0);
    y += translation(1);
    z += translation(2);
  }

  inline std::optional<Bond> is_bonded(const Atom &other,
                                       double bond_tolerance = 0.4) const {
    double rsq = this->square_distance(other);
    return is_bonded_with_sq_distance(other, rsq, bond_tolerance);
  }

  inline std::optional<Bond>
  is_bonded_with_rsq(const Atom &other, double rsq,
                     double bond_tolerance = 0.4) const {
    return is_bonded_with_sq_distance(other, rsq, bond_tolerance);
  }

  inline bool operator==(const Atom &rhs) const {
    return this->index == rhs.index;
  }

private:
  inline std::optional<Bond>
  is_bonded_with_sq_distance(const Atom &other, double rsq,
                             double bond_tolerance) const {
    double distance = std::sqrt(rsq);
    double bond_threshold = element.covalent_radius() +
                            other.element.covalent_radius() + bond_tolerance;
    if (distance > bond_threshold) {
      return std::nullopt;
    }
    Bond bond(distance);
    return bond;
  }
};
}; // namespace trajan::core
