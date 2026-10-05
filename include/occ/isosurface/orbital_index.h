#pragma once
#include <string>
#include <vector>

namespace occ::isosurface {

struct OrbitalIndex {
  enum class Reference { Absolute, HOMO, LUMO };

  int offset{0};
  Reference reference{Reference::Absolute};

  // 0-based MO index; HOMO/LUMO are relative to the occupied orbitals of
  // the spin in question (alpha, or beta for unrestricted beta orbitals)
  int resolve(int num_occupied) const;
  std::string format() const;
};

std::vector<OrbitalIndex> parse_orbital_descriptions(const std::string &input);

} // namespace occ::isosurface
