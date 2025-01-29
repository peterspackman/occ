#include <occ/core/log.h>
#include <occ/isosurface/stockholder_weight.h>
#include <occ/slater/slaterbasis.h>

namespace occ::isosurface {

StockholderWeightFunctor::StockholderWeightFunctor(
    const occ::core::Molecule &in, occ::core::Molecule &ext, float sep,
    const occ::slater::InterpolatorParams &params)
    : m_target_separation(sep),
      m_hirshfeld(occ::slater::PromoleculeDensity(in, params),
                  occ::slater::PromoleculeDensity(ext, params)) {

  const auto &interior_elements = in.atomic_numbers();
  size_t num_interior = interior_elements.rows();

  const auto &exterior_elements = ext.atomic_numbers();
  size_t num_exterior = exterior_elements.rows();

  Mat3N positions(3, num_interior + num_exterior);
  positions.block(0, 0, 3, num_interior) =
      in.positions().array() * occ::units::ANGSTROM_TO_BOHR;
  positions.block(0, num_interior, 3, num_exterior) =
      ext.positions().array() * occ::units::ANGSTROM_TO_BOHR;

  Eigen::Vector3f interior_m_minimum_atom_pos =
      positions.block(0, 0, 3, num_interior).rowwise().minCoeff().cast<float>();
  Eigen::Vector3f interior_m_maximum_atom_pos =
      positions.block(0, 0, 3, num_interior).rowwise().maxCoeff().cast<float>();

  m_origin = interior_m_minimum_atom_pos.array() - m_buffer;
  m_cube_side_length =
      (interior_m_maximum_atom_pos - m_origin).array() + m_buffer;

  occ::log::debug("Buffer region: {:.3f} bohr", m_buffer);
  occ::log::debug("Cube side length: {:.3f} {:.3f} {:.3f} bohr",
                  m_cube_side_length(0), m_cube_side_length(1),
                  m_cube_side_length(2));
  occ::log::debug("Target separation: {:.3f} bohr", m_target_separation);

  // set up bounding box to short cut if
  // we have a very anisotropic molecule
  m_bounding_box.lower = m_origin;
  m_bounding_box.upper = interior_m_maximum_atom_pos.cast<float>();
  m_bounding_box.upper.array() += m_buffer;

  occ::log::debug("Bottom left [{:.3f}, {:.3f}, {:.3f}]", m_origin(0),
                  m_origin(1), m_origin(2));
}

} // namespace occ::isosurface
