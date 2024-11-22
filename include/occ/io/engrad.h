#pragma once
#include <istream>
#include <occ/core/linear_algebra.h>

namespace occ::io {

class EngradReader {
public:
  EngradReader(const std::string &filename);
  EngradReader(std::istream &);

  inline const auto &positions() const { return m_positions; }
  inline const auto &gradient() const { return m_gradient; }
  inline int num_atoms() const { return m_num_atoms; }
  inline double energy() const { return m_energy; }
  inline const auto &atomic_numbers() const { return m_atomic_numbers; }

private:
  void parse(std::istream &);
  int m_num_atoms{0};
  double m_energy{0.0};
  Mat3N m_gradient;
  IVec m_atomic_numbers;
  Mat3N m_positions;
};
} // namespace occ::io
