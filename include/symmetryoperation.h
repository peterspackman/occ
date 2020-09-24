#pragma once
#include "linear_algebra.h"
#include <string>

namespace craso::crystal {

using craso::Mat3;
using craso::Mat3N;
using craso::Mat4;
using craso::Vec3;

class SymmetryOperation {
public:
  SymmetryOperation(const std::string &);
  SymmetryOperation(int);

  int to_int() const { return m_int; }
  void set_from_int(int);

  const std::string &to_string() const { return m_str; }
  void set_from_string(const std::string &);
  SymmetryOperation inverted() const;
  SymmetryOperation translated(const Vec3 &) const;
  bool is_identity() const { return m_int == 16484; }

  auto apply(const Mat3N &frac) const {
    Mat3N tmp = m_rotation * frac;
    tmp.colwise() += m_translation;
    return tmp;
  }

  const auto &seitz() const { return m_seitz; }
  const auto &rotation() const { return m_rotation; }
  const auto &translation() const { return m_translation; }

  // Operators
  auto operator()(const Mat3N &frac) const { return apply(frac); }
  bool operator==(const SymmetryOperation &other) const {
    return m_int == other.m_int;
  }
  bool operator<(const SymmetryOperation &other) const {
    return m_int < other.m_int;
  }
  bool operator>(const SymmetryOperation &other) const {
    return m_int > other.m_int;
  }
  bool operator<=(const SymmetryOperation &other) const {
    return m_int <= other.m_int;
  }
  bool operator>=(const SymmetryOperation &other) const {
    return m_int >= other.m_int;
  }

private:
  void update_from_seitz();
  int m_int;
  std::string m_str;
  Mat4 m_seitz;
  // above is the core data, this is just convenience
  Mat3 m_rotation;
  Vec3 m_translation;
};

} // namespace craso::crystal
