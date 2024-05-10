#include <occ/isosurface/electric_potential.h>
#include <occ/core/molecule.h>
#include <occ/core/log.h>

namespace occ::isosurface {

ElectricPotentialFunctor::ElectricPotentialFunctor(
    const occ::qm::Wavefunction &wfn, float sep) 
    : m_hf(wfn.basis), m_wfn(wfn), m_target_separation(sep) {

    core::Molecule mol(m_wfn.atoms);
    Mat3N coordinates = mol.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    m_minimum_atom_pos = coordinates.rowwise().minCoeff().cast<float>();
    m_maximum_atom_pos = coordinates.rowwise().maxCoeff().cast<float>();

    update_region();
}

void ElectricPotentialFunctor::update_region() {

    m_buffer = 5.0;

    m_origin = m_minimum_atom_pos.array() - m_buffer;

    m_cube_side_length =
        (m_maximum_atom_pos - m_origin).array() + m_buffer;

    occ::log::info("Buffer region: {:.3f} bohr", m_buffer);
    occ::log::info("Cube side lengths: [{:.3f} {:.3f} {:.3f}] bohr",
	    m_cube_side_length(0), m_cube_side_length(1), m_cube_side_length(2));
    occ::log::info("Target separation: {:.3f} bohr", m_target_separation);

    // set up bounding box to short cut if
    // we have a very anisotropic molecule
    m_bounding_box.lower = m_origin;
    m_bounding_box.upper = m_maximum_atom_pos;
    m_bounding_box.upper.array() += m_buffer;

    occ::log::info("Bottom left [{:.3f}, {:.3f}, {:.3f}]",
                   m_origin(0), m_origin(1), m_origin(2));
}

}
