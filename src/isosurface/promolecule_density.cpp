#include <occ/isosurface/promolecule_density.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/log.h>

namespace occ::isosurface {

PromoleculeDensityFunctor::PromoleculeDensityFunctor(
    const occ::core::Molecule &mol, float sep, const occ::slater::InterpolatorParams &params)
    : m_target_separation(sep), m_promol(mol, params) {

    FMat3N coordinates = (mol.positions().array() * occ::units::ANGSTROM_TO_BOHR).cast<float>();

    m_minimum_atom_pos = coordinates.rowwise().minCoeff();
    m_maximum_atom_pos = coordinates.rowwise().maxCoeff();

    update_region_for_isovalue();
}

void PromoleculeDensityFunctor::update_region_for_isovalue() {

    m_buffer = m_promol.maximum_distance_heuristic(m_isovalue, 1.0f);

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
