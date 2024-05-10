#include <occ/core/element.h>
#include <occ/crystal/asymmetric_unit.h>

namespace occ::crystal {

AsymmetricUnit::AsymmetricUnit(const Mat3N &frac_pos, const IVec &nums)
    : positions(frac_pos), atomic_numbers(nums), occupations(nums.rows()),
      charges(nums.rows()) {
    occupations.setConstant(1.0);
    charges = atomic_numbers.cast<double>();
    generate_default_labels();
}

AsymmetricUnit::AsymmetricUnit(const Mat3N &frac_pos, const IVec &nums,
                               const std::vector<std::string> &site_labels)
    : positions(frac_pos), atomic_numbers(nums), occupations(nums.rows()),
      charges(nums.rows()), labels(site_labels) {
    charges = atomic_numbers.cast<double>();
    occupations.setConstant(1.0);
}

void AsymmetricUnit::generate_default_labels() {
    occ::IVec counts(atomic_numbers.maxCoeff() + 1);
    counts.setConstant(1);
    labels.clear();
    labels.reserve(size());
    for (size_t i = 0; i < size(); i++) {
        auto num = atomic_numbers(i);
        auto symbol = occ::core::Element(num).symbol();
        labels.push_back(fmt::format("{}{}", symbol, counts(num)++));
    }
}

Eigen::VectorXd AsymmetricUnit::covalent_radii() const {
    Eigen::VectorXd result(atomic_numbers.size());
    for (int i = 0; i < atomic_numbers.size(); i++) {
        result(i) = occ::core::Element(atomic_numbers(i)).covalent_radius();
    }
    return result;
}

std::string AsymmetricUnit::chemical_formula() const {
    std::vector<occ::core::Element> els;
    for (int i = 0; i < atomic_numbers.size(); i++) {
        els.push_back(occ::core::Element(atomic_numbers[i]));
    }
    return occ::core::chemical_formula(els);
}

} // namespace occ::crystal
