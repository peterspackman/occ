#include <occ/isosurface/stockholder_weight.h>
#include <occ/core/log.h>
#include <occ/slater/slaterbasis.h>

namespace occ::isosurface {

StockholderWeightFunctor::StockholderWeightFunctor(
    const occ::core::Molecule &in, occ::core::Molecule &ext, float sep,
    const InterpolatorParams &params)
    : m_target_separation(sep), m_interpolator_params(params) {

    const auto &interior_elements = in.atomic_numbers();
    m_num_interior = interior_elements.rows();

    const auto &exterior_elements = ext.atomic_numbers();
    size_t num_exterior = exterior_elements.rows();

    Mat3N positions(3, m_num_interior + num_exterior);
    positions.block(0, 0, 3, m_num_interior) =
        in.positions().array() * occ::units::ANGSTROM_TO_BOHR;
    positions.block(0, m_num_interior, 3, num_exterior) =
        ext.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    Eigen::Vector3f interior_m_minimum_atom_pos =
        positions.block(0, 0, 3, m_num_interior)
            .rowwise()
            .minCoeff()
            .cast<float>();
    Eigen::Vector3f interior_m_maximum_atom_pos =
        positions.block(0, 0, 3, m_num_interior)
            .rowwise()
            .maxCoeff()
            .cast<float>();

    auto basis = occ::slater::load_slaterbasis("thakkar");
    ankerl::unordered_dense::map<int, std::vector<int>> tmp_map;

    auto maybe_create_interpolator = [&](int el) {
        auto search = m_interpolators.find(el);
        if (search == m_interpolators.end()) {
            auto b = basis[occ::core::Element(el).symbol()];
            auto func = [&b](float x) { return b.rho(std::sqrt(x)); };
            m_interpolators[el] = LinearInterpolatorFloat(
                func, m_interpolator_params.domain_lower,
                m_interpolator_params.domain_upper,
                m_interpolator_params.num_points);
        }
    };

    for (size_t i = 0; i < m_num_interior; i++) {
        int el = interior_elements(i);
        tmp_map[el].push_back(i);
        maybe_create_interpolator(el);
    }

    ankerl::unordered_dense::map<int, int> interior_counts;
    for (const auto &kv : tmp_map) {
        interior_counts[kv.first] = kv.second.size();
    }

    for (size_t i = 0; i < num_exterior; i++) {
        int el = exterior_elements(i);
        tmp_map[el].push_back(m_num_interior + i);
        maybe_create_interpolator(el);
    }

    for (const auto &kv : tmp_map) {
        m_atom_interpolators.push_back(
            {m_interpolators.at(kv.first),
             positions(Eigen::all, kv.second).cast<float>(), 0.0,
             interior_counts[kv.first]});
        m_atom_interpolators.back().threshold =
            m_atom_interpolators.back().interpolator.find_threshold(1e-8);
    }

    m_origin = interior_m_minimum_atom_pos.array() - m_buffer;
    m_cube_side_length =
        (interior_m_maximum_atom_pos - m_origin).array() + m_buffer;

    occ::log::info("Buffer region: {:.3f} bohr", m_buffer);
    occ::log::info("Cube side length: {:.3f} {:.3f} {:.3f} bohr", m_cube_side_length(0), m_cube_side_length(1), m_cube_side_length(2));
    occ::log::info("Target separation: {:.3f} bohr", m_target_separation);


    // set up bounding box to short cut if
    // we have a very anisotropic molecule
    m_bounding_box.lower = m_origin;
    m_bounding_box.upper = interior_m_maximum_atom_pos.cast<float>();
    m_bounding_box.upper.array() += m_buffer;

    occ::log::info("Bottom left [{:.3f}, {:.3f}, {:.3f}]",
                   m_origin(0), m_origin(1), m_origin(2));
}


}
