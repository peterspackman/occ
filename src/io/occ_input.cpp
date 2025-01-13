#include <occ/core/atom.h>
#include <occ/core/units.h>
#include <occ/io/occ_input.h>

namespace occ::io {

occ::core::Molecule GeometryInput::molecule() const {
  return occ::core::Molecule(elements, positions);
}

void GeometryInput::set_molecule(const occ::core::Molecule &mol) {
  elements = mol.elements();
  positions.clear();
  const auto &pos = mol.positions();
  positions.reserve(pos.cols());
  for (size_t i = 0; i < elements.size(); i++) {
    positions.push_back(std::array<double, 3>{pos(0, i), pos(1, i), pos(2, i)});
  }
}

} // namespace occ::io
