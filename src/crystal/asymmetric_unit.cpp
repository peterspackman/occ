#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/crystal/asymmetric_unit.h>

namespace occ::crystal {

using core::Element;

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
  IVec counts(atomic_numbers.maxCoeff() + 1);
  counts.setConstant(1);
  labels.clear();
  labels.reserve(size());
  for (size_t i = 0; i < size(); i++) {
    auto num = atomic_numbers(i);
    auto symbol = Element(num).symbol();
    labels.push_back(fmt::format("{}{}", symbol, counts(num)++));
  }
}

Vec AsymmetricUnit::covalent_radii() const {
  Eigen::VectorXd result(atomic_numbers.size());
  for (int i = 0; i < atomic_numbers.size(); i++) {
    result(i) = Element(atomic_numbers(i)).covalent_radius();
  }
  return result;
}

Vec AsymmetricUnit::vdw_radii() const {
  Eigen::VectorXd result(atomic_numbers.size());
  for (int i = 0; i < atomic_numbers.size(); i++) {
    result(i) = Element(atomic_numbers(i)).van_der_waals_radius();
  }
  return result;
}

void AsymmetricUnit::resize(Eigen::Index n) {
  positions.resize(3, n);
  atomic_numbers.resize(n);
  occupations = Vec::Ones(n);
  charges = Vec::Zero(n);
  adps = Mat6N::Zero(6, n);
  labels.clear();
}

bool AsymmetricUnit::is_consistent() const {
  const Eigen::Index n = atomic_numbers.size();
  return positions.cols() == n && occupations.size() == n &&
         charges.size() == n && adps.cols() == n &&
         static_cast<Eigen::Index>(labels.size()) == n;
}

void AsymmetricUnit::ensure_consistent() {
  const Eigen::Index n = atomic_numbers.size();
  // Only fill in what is missing -- never overwrite data that is already there.
  if (occupations.size() != n)
    occupations = Vec::Ones(n);
  if (charges.size() != n)
    charges = Vec::Zero(n);
  if (adps.cols() != n)
    adps = Mat6N::Zero(6, n);
  if (static_cast<Eigen::Index>(labels.size()) != n)
    generate_default_labels();
}

std::string AsymmetricUnit::chemical_formula() const {
  std::vector<Element> els;
  for (int i = 0; i < atomic_numbers.size(); i++) {
    els.push_back(Element(atomic_numbers[i]));
  }
  return core::chemical_formula(els);
}

} // namespace occ::crystal
