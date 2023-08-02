#include <occ/core/log.h>
#include <occ/main/isosurface.h>

namespace occ::main {

StockholderWeightFunctor::StockholderWeightFunctor(
    const occ::core::Molecule &in, occ::core::Molecule &ext, float sep,
    const InterpolatorParams &params)
    : m_target_separation(sep), m_interpolator_params(params) {

    const auto &interior_elements = in.atomic_numbers();
    Mat3N interior_coordinates =
        in.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    const auto &exterior_elements = ext.atomic_numbers();
    Mat3N exterior_coordinates =
        ext.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    Eigen::Vector3f interior_minimum_pos =
        interior_coordinates.rowwise().minCoeff().cast<float>();
    Eigen::Vector3f interior_maximum_pos =
        interior_coordinates.rowwise().maxCoeff().cast<float>();

    auto basis = occ::slater::load_slaterbasis("thakkar");
    phmap::flat_hash_map<int, std::vector<int>> tmp_map;
    for (size_t i = 0; i < in.size(); i++) {
        int el = interior_elements(i);
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
        m_interior.push_back(
            {m_interpolators.at(kv.first),
             interior_coordinates(Eigen::all, kv.second).cast<float>()});
        m_interior.back().threshold =
            m_interior.back().interpolator.find_threshold(1e-8);
    }

    tmp_map.clear();
    for (size_t i = 0; i < ext.size(); i++) {
        int el = exterior_elements(i);
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
        m_exterior.push_back(
            {m_interpolators.at(kv.first),
             exterior_coordinates(Eigen::all, kv.second).cast<float>()});
        m_exterior.back().threshold =
            m_exterior.back().interpolator.find_threshold(1e-8);
    }

    m_origin = interior_minimum_pos.array() - m_buffer;
    m_cube_side_length =
        (interior_maximum_pos - m_origin).maxCoeff() + m_buffer;

    m_subdivisions = std::ceil(
        std::log(m_cube_side_length / m_target_separation) / std::log(2));

    double new_m_cube_side_length =
        m_target_separation * std::pow(2, m_subdivisions);
    occ::log::debug("Cube side length: {}", m_cube_side_length);
    occ::log::debug("Target separation: {}", m_target_separation);
    occ::log::debug("Suggested side m_cube_side_length: {}",
                    new_m_cube_side_length);
    occ::log::debug("Subdivisions: {} (cube size = {})\n", m_subdivisions,
                    new_m_cube_side_length / std::pow(2, m_subdivisions));
    m_cube_side_length = new_m_cube_side_length;

    // set up bounding box to short cut if
    // we have a very anisotropic molecule
    m_bounding_box.lower = m_origin;
    m_bounding_box.upper = interior_maximum_pos.cast<float>();
    m_bounding_box.upper.array() += m_buffer;

    // we have a scale m_diagonal_scale_factortor as the cubes are unit cubes
    // i.e. their diagonals are sqrt(3)
    //
    m_diagonal_scale_factor = 1.0 / m_cube_side_length;
    occ::log::debug("Bottom left [{:.3f}, {:.3f}, {:.3f}], side length = {}",
                    m_origin(0), m_origin(1), m_origin(2), m_cube_side_length);
}

PromoleculeDensityFunctor::PromoleculeDensityFunctor(
    const occ::core::Molecule &mol, float sep, const InterpolatorParams &params)
    : m_target_separation(sep), m_interpolator_params(params) {

    const auto &elements = mol.atomic_numbers();
    Mat3N coordinates = mol.positions().array() * occ::units::ANGSTROM_TO_BOHR;

    Eigen::Vector3f minimum_pos =
        coordinates.rowwise().minCoeff().cast<float>();
    Eigen::Vector3f maximum_pos =
        coordinates.rowwise().maxCoeff().cast<float>();

    auto basis = occ::slater::load_slaterbasis("thakkar");
    phmap::flat_hash_map<int, std::vector<int>> tmp_map;
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

    m_origin = minimum_pos.array() - m_buffer;
    m_cube_side_length = (maximum_pos - m_origin).maxCoeff() + m_buffer;

    m_subdivisions = std::ceil(
        std::log(m_cube_side_length / m_target_separation) / std::log(2));

    double new_m_cube_side_length =
        m_target_separation * std::pow(2, m_subdivisions);
    occ::log::debug("Cube side length: {}", m_cube_side_length);
    occ::log::debug("Target separation: {}", m_target_separation);
    occ::log::debug("Suggested side m_cube_side_length: {}",
                    new_m_cube_side_length);
    occ::log::debug("Subdivisions: {} (cube size = {})", m_subdivisions,
                    new_m_cube_side_length / std::pow(2, m_subdivisions));
    m_cube_side_length = new_m_cube_side_length;

    // set up bounding box to short cut if
    // we have a very anisotropic molecule
    m_bounding_box.lower = m_origin;
    m_bounding_box.upper = maximum_pos.cast<float>();
    m_bounding_box.upper.array() += m_buffer;

    // we have a scale m_diagonal_scale_factortor as the cubes are unit cubes
    // i.e. their diagonals are sqrt(3)
    //
    m_diagonal_scale_factor = 1.0 / m_cube_side_length;
    occ::log::debug("Bottom left [{:.3f}, {:.3f}, {:.3f}], side length = {}",
                    m_origin(0), m_origin(1), m_origin(2), m_cube_side_length);
}

} // namespace occ::main
