#include <occ/isosurface/promolecule_density.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/log.h>

namespace occ::isosurface {

PromoleculeDensityFunctor::PromoleculeDensityFunctor(
    const occ::core::Molecule &mol, float sep, const InterpolatorParams &params)
    : m_target_separation(sep), m_interpolator_params(params) {

    const auto &elements = mol.atomic_numbers();
    Mat3N coordinates = mol.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    m_minimum_atom_pos = coordinates.rowwise().minCoeff().cast<float>();
    m_maximum_atom_pos = coordinates.rowwise().maxCoeff().cast<float>();

    auto basis = occ::slater::load_slaterbasis("thakkar");
    ankerl::unordered_dense::map<int, std::vector<int>> tmp_map;
    for (size_t i = 0; i < elements.rows(); i++) {
        int el = elements(i);
        tmp_map[el].push_back(i);
        auto search = m_interpolators.find(el);
        if (search == m_interpolators.end()) {
            auto b = basis[occ::core::Element(el).symbol()];
            auto func = [&b](float x) { return b.rho(std::sqrt(x)); };
            m_interpolators[el] = LinearInterpolatorFloat(
                func, m_interpolator_params.domain_lower,
                m_interpolator_params.domain_upper,
                m_interpolator_params.num_points);
        }
    }

    for (const auto &kv : tmp_map) {
        m_atom_interpolators.push_back(
            {m_interpolators.at(kv.first),
             coordinates(Eigen::all, kv.second).cast<float>()});
        m_atom_interpolators.back().threshold =
            m_atom_interpolators.back().interpolator.find_threshold(1e-8);
    }

    update_region_for_isovalue();
}

void PromoleculeDensityFunctor::update_region_for_isovalue() {

    for (const auto &[interp, coords, threshold, interior] :
         m_atom_interpolators) {
        m_buffer = std::max(interp.find_threshold(m_isovalue) + 1.0f, m_buffer);
    }

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
