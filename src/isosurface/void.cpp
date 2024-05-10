#include <occ/isosurface/void.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/log.h>

namespace occ::isosurface {

VoidSurfaceFunctor::VoidSurfaceFunctor(
	const occ::crystal::Crystal &crystal,
	float sep, const InterpolatorParams &params) : 
    m_crystal(crystal), 
    m_interpolator_params(params),
    m_target_separation(sep) {

    double radius = m_buffer;
    const auto &uc_atoms = m_crystal.unit_cell_atoms();
    crystal::HKL upper = crystal::HKL::minimum();
    crystal::HKL lower = crystal::HKL::maximum();
    occ::Vec3 frac_radius = radius * 2 / m_crystal.unit_cell().lengths().array();

    for (size_t i = 0; i < uc_atoms.frac_pos.cols(); i++) {
        const auto &pos = uc_atoms.frac_pos.col(i);
        upper.h =
            std::max(upper.h, static_cast<int>(ceil(pos(0) + frac_radius(0))));
        upper.k =
            std::max(upper.k, static_cast<int>(ceil(pos(1) + frac_radius(1))));
        upper.l =
            std::max(upper.l, static_cast<int>(ceil(pos(2) + frac_radius(2))));

        lower.h =
            std::min(lower.h, static_cast<int>(floor(pos(0) - frac_radius(0))));
        lower.k =
            std::min(lower.k, static_cast<int>(floor(pos(1) - frac_radius(1))));
        lower.l =
            std::min(lower.l, static_cast<int>(floor(pos(2) - frac_radius(2))));
    }
    auto slab = m_crystal.slab(lower, upper);

    const auto &elements = slab.atomic_numbers;
    Mat3N coordinates = slab.cart_pos.array() * occ::units::ANGSTROM_TO_BOHR;

    m_molecule = occ::core::Molecule(elements, slab.cart_pos);

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

    update_region();
}

void VoidSurfaceFunctor::update_region() {

    m_origin.setConstant(0);
    Eigen::Vector3f max_pos =
        m_crystal.unit_cell().direct()
            .rowwise()
            .maxCoeff()
            .cast<float>();
    m_cube_side_length = max_pos * occ::units::ANGSTROM_TO_BOHR;

    occ::log::info("Cube side lengths: [{:.3f} {:.3f} {:.3f}] bohr",
	    m_cube_side_length(0), m_cube_side_length(1), m_cube_side_length(2));
    occ::log::info("Target separation: {:.3f} bohr", m_target_separation);
}


}
