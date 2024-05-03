#include <occ/slater/promolecule.h>
#include <occ/slater/slaterbasis.h>
#include <occ/core/log.h>

namespace occ::slater {

PromoleculeDensity::PromoleculeDensity(const occ::core::Molecule &mol, const InterpolatorParams &params)
    : m_interpolator_params(params) {

    const auto &elements = mol.atomic_numbers();
    FMat3N positions = (mol.positions().array() * occ::units::ANGSTROM_TO_BOHR).cast<float>();
    initialize_interpolators(elements, positions);
}

void PromoleculeDensity::initialize_interpolators(Eigen::Ref<const IVec> elements,
						  Eigen::Ref<const FMat3N> positions) {

    occ::log::debug("Loading slater basis 'thakkar'");
    auto basis = occ::slater::load_slaterbasis("thakkar");

    ankerl::unordered_dense::map<int, impl::Interpolator> interpolators;
    ankerl::unordered_dense::map<int, std::vector<int>> tmp_map;

    for (size_t i = 0; i < elements.rows(); i++) {
        int el = elements(i);
        tmp_map[el].push_back(i);
        auto search = interpolators.find(el);
        if (search == interpolators.end()) {
            auto b = basis[occ::core::Element(el).symbol()];
            auto func = [&b](float x) { return b.rho(std::sqrt(x)); };
            interpolators[el] = impl::Interpolator(
                func, m_interpolator_params.domain_lower,
                m_interpolator_params.domain_upper,
                m_interpolator_params.num_points);
        }
    }

    for (const auto &kv : tmp_map) {
        m_atom_interpolators.push_back(
            {interpolators.at(kv.first),
             positions(Eigen::all, kv.second)});
        m_atom_interpolators.back().threshold =
            m_atom_interpolators.back().interpolator.find_threshold(1e-8);
    }
}

float PromoleculeDensity::maximum_distance_heuristic(float value, float buffer) const {
    float result{0.0};

    for (const auto &[interp, coords, threshold] :
         m_atom_interpolators) {
        result = std::max(interp.find_threshold(value) + buffer, result);
    }
    return result;
}

}
