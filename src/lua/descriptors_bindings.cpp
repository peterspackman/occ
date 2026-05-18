#include "descriptors_bindings.h"
#include "eigen_conv.h"
#include <occ/crystal/crystal.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/promolecule_shape.h>
#include <occ/descriptors/steinhardt.h>

namespace sol {
template <>
struct is_automagical<occ::descriptors::PointwiseDistanceDistribution>
    : std::false_type {};
template <>
struct is_automagical<occ::descriptors::Steinhardt> : std::false_type {};
template <>
struct is_container<occ::descriptors::PointwiseDistanceDistribution>
    : std::false_type {};
template <>
struct is_container<occ::descriptors::Steinhardt> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::descriptors;
using occ::crystal::Crystal;

void register_descriptors_bindings(sol::state_view, sol::table &m) {
  m.new_usertype<PointwiseDistanceDistributionConfig>(
      "PDDConfig",
      sol::call_constructor,
      sol::factories([]() { return PointwiseDistanceDistributionConfig{}; }),
      "lexsort", &PointwiseDistanceDistributionConfig::lexsort,
      "collapse", &PointwiseDistanceDistributionConfig::collapse,
      "collapse_tol", &PointwiseDistanceDistributionConfig::collapse_tol,
      "return_groups",
      &PointwiseDistanceDistributionConfig::return_groups);

  m.new_usertype<PointwiseDistanceDistribution>(
      "PDD",
      sol::call_constructor,
      sol::factories(
          [](const Crystal &c, int k) {
            return PointwiseDistanceDistribution(c, k);
          },
          [](const Crystal &c, int k,
             const PointwiseDistanceDistributionConfig &cfg) {
            return PointwiseDistanceDistribution(c, k, cfg);
          }),
      "weights",
      sol::readonly_property(
          [](const PointwiseDistanceDistribution &p, sol::this_state s) {
            return p.weights();
          }),
      "distances",
      sol::readonly_property(
          [](const PointwiseDistanceDistribution &p, sol::this_state s) {
            return p.distances();
          }),
      "average_minimum_distance",
      &PointwiseDistanceDistribution::average_minimum_distance,
      "matrix",
      [](const PointwiseDistanceDistribution &p, sol::this_state s) {
        return p.matrix();
      },
      "size", &PointwiseDistanceDistribution::size,
      "k", &PointwiseDistanceDistribution::k,
      // groups() returns Eigen::MatrixXi (int matrix), not a std::vector —
      // use mat_to_table, which handles any Eigen::MatrixBase.
      "groups",
      sol::readonly_property(
          [](const PointwiseDistanceDistribution &p, sol::this_state s) {
            return p.groups();
          }));

  m.new_usertype<Steinhardt>(
      "Steinhardt",
      sol::call_constructor, sol::constructors<Steinhardt(size_t)>(),
      "compute_q",
      [](Steinhardt &st, const sol::table &positions, sol::this_state s) {
        return st.compute_q(table_to_mat3n(positions));
      },
      "compute_w",
      [](Steinhardt &st, const sol::table &positions, sol::this_state s) {
        return st.compute_w(table_to_mat3n(positions));
      },
      "compute_averaged_q",
      [](Steinhardt &st, const sol::table &positions,
         sol::optional<double> radius, sol::this_state s) {
        return vec_to_table(s, st.compute_averaged_q(table_to_mat3n(positions),
                                                       radius.value_or(6.0)));
      },
      "compute_averaged_w",
      [](Steinhardt &st, const sol::table &positions,
         sol::optional<double> radius, sol::this_state s) {
        return vec_to_table(s, st.compute_averaged_w(table_to_mat3n(positions),
                                                       radius.value_or(6.0)));
      },
      "precompute_wigner3j_coefficients",
      &Steinhardt::precompute_wigner3j_coefficients,
      "size", &Steinhardt::size,
      "nlm", &Steinhardt::nlm);

  m.new_usertype<PromoleculeDensityShape::InterpolatorParameters>(
      "PromoleculeInterpolatorParameters",
      sol::call_constructor,
      sol::factories(
          []() { return PromoleculeDensityShape::InterpolatorParameters{}; }),
      "num_points",
      &PromoleculeDensityShape::InterpolatorParameters::num_points,
      "domain_lower",
      &PromoleculeDensityShape::InterpolatorParameters::domain_lower,
      "domain_upper",
      &PromoleculeDensityShape::InterpolatorParameters::domain_upper);
}

} // namespace occ::lua_bindings
