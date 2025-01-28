#pragma once
#include <string>
#include <vector>

namespace occ::isosurface {

struct OrbitalIndex {
  enum class Reference { Absolute, HOMO, LUMO };

  int offset{0};
  Reference reference{Reference::Absolute};

  int resolve(int nalpha, int nbeta) const;
  std::string format() const;
};

std::vector<OrbitalIndex> parse_orbital_descriptions(const std::string &input);

} // namespace occ::isosurface
