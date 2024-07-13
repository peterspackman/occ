#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/isosurface/eeq_esp.h>

namespace occ::isosurface {

ElectricPotentialFunctorPC::ElectricPotentialFunctorPC(
    const occ::core::Molecule &m)
    : m_molecule(m) {}

} // namespace occ::isosurface
