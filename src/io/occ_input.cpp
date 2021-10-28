#include <occ/io/occ_input.h>
#include <occ/core/atom.h>
#include <occ/core/units.h>

namespace occ::io {

occ::chem::Molecule GeometryInput::molecule() const {
    return occ::chem::Molecule(elements, positions);
}

} // namespace occ::io
