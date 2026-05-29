#include "descriptors_bindings.h"
#include "eigen_conv.h"
#include <occ/crystal/crystal.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/promolecule_shape.h>
#include <occ/descriptors/steinhardt.h>

namespace occ::lua_bindings {

using namespace occ::descriptors;
using occ::crystal::Crystal;
namespace lb = luabridge;

void register_descriptors_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<PointwiseDistanceDistributionConfig>("PDDConfig")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("lexsort",
                            &PointwiseDistanceDistributionConfig::lexsort)
      .addPropertyReadWrite("collapse",
                            &PointwiseDistanceDistributionConfig::collapse)
      .addPropertyReadWrite("collapse_tol",
                            &PointwiseDistanceDistributionConfig::collapse_tol)
      .addPropertyReadWrite("return_groups",
                            &PointwiseDistanceDistributionConfig::return_groups)
      .endClass()

      .beginClass<PointwiseDistanceDistribution>("PDD")
      // Two construction shapes — split into a canonical constructor
      // and a static factory for the configured variant.
      .addConstructor<void (*)(const Crystal &, int)>()
      .addStaticFunction(
          "new_with_config",
          +[](const Crystal &c, int k,
              const PointwiseDistanceDistributionConfig &cfg) {
            return new PointwiseDistanceDistribution(c, k, cfg);
          })
      .addProperty(
          "weights",
          +[](const PointwiseDistanceDistribution *p) -> occ::Vec {
            return p->weights();
          })
      .addProperty(
          "distances",
          +[](const PointwiseDistanceDistribution *p) -> occ::Mat {
            return p->distances();
          })
      .addProperty("average_minimum_distance",
                   &PointwiseDistanceDistribution::average_minimum_distance)
      .addProperty(
          "matrix",
          +[](const PointwiseDistanceDistribution *p) -> occ::Mat {
            return p->matrix();
          })
      .addProperty("size", &PointwiseDistanceDistribution::size)
      .addProperty("k", &PointwiseDistanceDistribution::k)
      .addProperty(
          "groups", +[](const PointwiseDistanceDistribution *p) -> occ::IMat {
            return p->groups();
          })
      .endClass()

      .beginClass<Steinhardt>("Steinhardt")
      .addConstructor<void (*)(size_t)>()
      .addFunction(
          "compute_q",
          +[](Steinhardt *st, const lb::LuaRef &positions, lua_State *S) {
            return vec_to_table(S, st->compute_q(table_to_mat3n(positions)));
          })
      .addFunction(
          "compute_w",
          +[](Steinhardt *st, const lb::LuaRef &positions, lua_State *S) {
            return vec_to_table(S, st->compute_w(table_to_mat3n(positions)));
          })
      // sol2's optional<double> radius default = 6.0 — split into two
      // explicit named functions.
      .addFunction(
          "compute_averaged_q",
          +[](Steinhardt *st, const lb::LuaRef &positions, double radius,
              lua_State *S) {
            return vec_to_table(
                S, st->compute_averaged_q(table_to_mat3n(positions), radius));
          })
      .addFunction(
          "compute_averaged_q_default",
          +[](Steinhardt *st, const lb::LuaRef &positions, lua_State *S) {
            return vec_to_table(
                S, st->compute_averaged_q(table_to_mat3n(positions), 6.0));
          })
      .addFunction(
          "compute_averaged_w",
          +[](Steinhardt *st, const lb::LuaRef &positions, double radius,
              lua_State *S) {
            return vec_to_table(
                S, st->compute_averaged_w(table_to_mat3n(positions), radius));
          })
      .addFunction(
          "compute_averaged_w_default",
          +[](Steinhardt *st, const lb::LuaRef &positions, lua_State *S) {
            return vec_to_table(
                S, st->compute_averaged_w(table_to_mat3n(positions), 6.0));
          })
      .addFunction("precompute_wigner3j_coefficients",
                   &Steinhardt::precompute_wigner3j_coefficients)
      .addProperty("size", &Steinhardt::size)
      .addProperty("nlm", &Steinhardt::nlm)
      .endClass()

      .beginClass<PromoleculeDensityShape::InterpolatorParameters>(
          "PromoleculeInterpolatorParameters")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite(
          "num_points",
          &PromoleculeDensityShape::InterpolatorParameters::num_points)
      .addPropertyReadWrite(
          "domain_lower",
          &PromoleculeDensityShape::InterpolatorParameters::domain_lower)
      .addPropertyReadWrite(
          "domain_upper",
          &PromoleculeDensityShape::InterpolatorParameters::domain_upper)
      .endClass()

      .endNamespace();
}

} // namespace occ::lua_bindings
