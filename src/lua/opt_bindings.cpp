#include "opt_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/optimization_state.h>

namespace occ::lua_bindings {

using namespace occ::opt;
namespace lb = luabridge;

void register_opt_bindings(lua_State *L) {
  // Python uses a submodule (`occpy.opt.X`); mirror that as a nested
  // namespace (`occ.opt.X`).
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginNamespace("opt")

          .beginNamespace("BondCoordinateType")
            .addProperty("COVALENT",
                         +[]() {
                           return static_cast<int>(
                               BondCoordinate::Type::COVALENT);
                         })
            .addProperty(
                "VDW",
                +[]() { return static_cast<int>(BondCoordinate::Type::VDW); })
          .endNamespace()

          .beginClass<BondCoordinate>("BondCoordinate")
            // Two construction shapes — default COVALENT vs explicit type.
            .addConstructor<void (*)(int, int, BondCoordinate::Type)>()
            .addStaticFunction(
                "new_covalent",
                +[](int i, int j) {
                  return new BondCoordinate{i, j,
                                              BondCoordinate::Type::COVALENT};
                })
            .addPropertyReadWrite("i", &BondCoordinate::i)
            .addPropertyReadWrite("j", &BondCoordinate::j)
            .addPropertyReadWrite("bond_type", &BondCoordinate::bond_type)
            .addFunction("value",
                         +[](const BondCoordinate *b,
                             const lb::LuaRef &coords) {
                           return (*b)(table_to_mat3n(coords));
                         })
            .addFunction("gradient",
                         +[](const BondCoordinate *b, const lb::LuaRef &coords,
                             lua_State *S) {
                           return mat_to_table(
                               S, b->gradient(table_to_mat3n(coords)));
                         })
            .addFunction("__tostring", +[](const BondCoordinate *b) {
              const char *type = (b->bond_type == BondCoordinate::Type::COVALENT)
                                     ? "COVALENT"
                                     : "VDW";
              return fmt::format("<BondCoordinate({}, {}, {})>", b->i, b->j,
                                 type);
            })
          .endClass()

          .beginClass<AngleCoordinate>("AngleCoordinate")
            .addConstructor<void (*)(int, int, int)>()
            .addPropertyReadWrite("i", &AngleCoordinate::i)
            .addPropertyReadWrite("j", &AngleCoordinate::j)
            .addPropertyReadWrite("k", &AngleCoordinate::k)
            .addFunction("value",
                         +[](const AngleCoordinate *a,
                             const lb::LuaRef &coords) {
                           return (*a)(table_to_mat3n(coords));
                         })
            .addFunction("gradient",
                         +[](const AngleCoordinate *a, const lb::LuaRef &coords,
                             lua_State *S) {
                           return mat_to_table(
                               S, a->gradient(table_to_mat3n(coords)));
                         })
            .addFunction("__tostring", +[](const AngleCoordinate *a) {
              return fmt::format("<AngleCoordinate({}, {}, {})>", a->i, a->j,
                                 a->k);
            })
          .endClass()

          .beginClass<DihedralCoordinate>("DihedralCoordinate")
            .addConstructor<void (*)(int, int, int, int)>()
            .addPropertyReadWrite("i", &DihedralCoordinate::i)
            .addPropertyReadWrite("j", &DihedralCoordinate::j)
            .addPropertyReadWrite("k", &DihedralCoordinate::k)
            .addPropertyReadWrite("l", &DihedralCoordinate::l)
            .addFunction("value",
                         +[](const DihedralCoordinate *d,
                             const lb::LuaRef &coords) {
                           return (*d)(table_to_mat3n(coords));
                         })
            .addFunction(
                "gradient",
                +[](const DihedralCoordinate *d, const lb::LuaRef &coords,
                    lua_State *S) {
                  return mat_to_table(S,
                                       d->gradient(table_to_mat3n(coords)));
                })
            .addFunction("__tostring", +[](const DihedralCoordinate *d) {
              return fmt::format("<DihedralCoordinate({}, {}, {}, {})>", d->i,
                                 d->j, d->k, d->l);
            })
          .endClass()

          .beginClass<InternalCoordinates::Options>(
              "InternalCoordinatesOptions")
            .addConstructor<void (*)()>()
            .addPropertyReadWrite(
                "include_dihedrals",
                &InternalCoordinates::Options::include_dihedrals)
            .addPropertyReadWrite(
                "superweak_dihedrals",
                &InternalCoordinates::Options::superweak_dihedrals)
          .endClass()

          .beginClass<InternalCoordinates>("InternalCoordinates")
            // Two construction shapes — molecule-only (defaults to
            // {include_dihedrals=true, superweak_dihedrals=false}) vs
            // options-configured.
            .addConstructor<void (*)(const occ::core::Molecule &,
                                      const InternalCoordinates::Options &)>()
            .addStaticFunction(
                "new_default",
                +[](const occ::core::Molecule &m) {
                  return new InternalCoordinates(
                      m, InternalCoordinates::Options{true, false});
                })
            .addFunction("bonds",
                         +[](const InternalCoordinates *ic, lua_State *S) {
                           lb::LuaRef t = lb::newTable(S);
                           const auto &bs = ic->bonds();
                           for (size_t i = 0; i < bs.size(); ++i) {
                             t[static_cast<int>(i + 1)] = bs[i];
                           }
                           return t;
                         })
            .addFunction("angles",
                         +[](const InternalCoordinates *ic, lua_State *S) {
                           lb::LuaRef t = lb::newTable(S);
                           const auto &as = ic->angles();
                           for (size_t i = 0; i < as.size(); ++i) {
                             t[static_cast<int>(i + 1)] = as[i];
                           }
                           return t;
                         })
            .addFunction("dihedrals",
                         +[](const InternalCoordinates *ic, lua_State *S) {
                           lb::LuaRef t = lb::newTable(S);
                           const auto &ds = ic->dihedrals();
                           for (size_t i = 0; i < ds.size(); ++i) {
                             t[static_cast<int>(i + 1)] = ds[i];
                           }
                           return t;
                         })
            .addProperty("weights",
                         +[](const InternalCoordinates *ic) -> occ::Vec { return ic->weights(); })
            .addProperty("size", &InternalCoordinates::size)
            .addFunction("__len",
                         +[](const InternalCoordinates *ic) {
                           return static_cast<int>(ic->size());
                         })
            .addFunction("to_vector",
                         +[](const InternalCoordinates *ic,
                             const lb::LuaRef &positions, lua_State *S) {
                           return vec_to_table(
                               S, ic->to_vector(table_to_mat3n(positions)));
                         })
            .addFunction(
                "to_vector_with_template",
                +[](const InternalCoordinates *ic, const lb::LuaRef &positions,
                    const lb::LuaRef &template_q, lua_State *S) {
                  return vec_to_table(
                      S, ic->to_vector_with_template(
                              table_to_mat3n(positions),
                              table_to_vecx(template_q)));
                })
            .addFunction(
                "wilson_b_matrix",
                +[](const InternalCoordinates *ic, const lb::LuaRef &positions,
                    lua_State *S) {
                  return mat_to_table(
                      S, ic->wilson_b_matrix(table_to_mat3n(positions)));
                })
            .addFunction("__tostring", +[](const InternalCoordinates *ic) {
              return fmt::format(
                  "<InternalCoordinates: {} bonds, {} angles, {} dihedrals>",
                  ic->bonds().size(), ic->angles().size(),
                  ic->dihedrals().size());
            })
          .endClass()

          .addFunction(
              "transform_step_to_cartesian",
              +[](const lb::LuaRef &internal_step,
                  const InternalCoordinates &coords,
                  const lb::LuaRef &positions, const lb::LuaRef &B_inv,
                  lua_State *S) {
                const int rows = B_inv.length();
                lb::LuaRef first_row = lua_get_table(B_inv, 1);
                const int cols = first_row.length();
                Mat Binv(rows, cols);
                for (int i = 0; i < rows; ++i) {
                  lb::LuaRef row = lua_get_table(B_inv, i + 1);
                  for (int j = 0; j < cols; ++j) {
                    Binv(i, j) = lua_get_num(row, j + 1);
                  }
                }
                return mat_to_table(
                    S, occ::opt::transform_step_to_cartesian(
                            table_to_vecx(internal_step), coords,
                            table_to_mat3n(positions), Binv));
              })

          .beginClass<OptPoint>("OptPoint")
            // Two construction shapes — default-empty vs full-init from
            // tables. The full-init takes Lua tables so split into a static
            // factory.
            .addConstructor<void (*)()>()
            .addStaticFunction(
                "new_from_tables",
                +[](const lb::LuaRef &q, double energy,
                    const lb::LuaRef &gradient) {
                  return new OptPoint{table_to_vecx(q), energy,
                                       table_to_vecx(gradient)};
                })
            .addProperty("get_q",
                         +[](const OptPoint *p) -> occ::Vec { return p->q; })
            .addFunction("set_q",
                         +[](OptPoint *p, const lb::LuaRef &t) {
                           p->q = table_to_vecx(t);
                         })
            .addPropertyReadWrite("E", &OptPoint::E)
            .addProperty("get_g",
                         +[](const OptPoint *p) -> occ::Vec { return p->g; })
            .addFunction("set_g",
                         +[](OptPoint *p, const lb::LuaRef &t) {
                           p->g = table_to_vecx(t);
                         })
            .addFunction("__tostring", +[](const OptPoint *p) {
              return fmt::format(
                  "<OptPoint: E={:.8f}, |q|={}, |g|={:.6f}>", p->E,
                  p->q.size(), p->g.size() > 0 ? p->g.norm() : 0.0);
            })
          .endClass()

          .beginClass<ConvergenceCriteria>("ConvergenceCriteria")
            .addConstructor<void (*)()>()
            .addPropertyReadWrite("gradient_max",
                                  &ConvergenceCriteria::gradient_max)
            .addPropertyReadWrite("gradient_rms",
                                  &ConvergenceCriteria::gradient_rms)
            .addPropertyReadWrite("step_max", &ConvergenceCriteria::step_max)
            .addPropertyReadWrite("step_rms", &ConvergenceCriteria::step_rms)
          .endClass()

          // OptimizationState exposes Eigen-backed fields — wrap each via a
          // function so we hand back a Lua table.
          .beginClass<OptimizationState>("OptimizationState")
            .addProperty("positions",
                         +[](const OptimizationState *o) -> occ::Mat3N { return o->positions; })
            .addProperty("energy", &OptimizationState::energy)
            .addProperty("gradient_cartesian",
                         +[](const OptimizationState *o) -> occ::Mat3N { return o->gradient_cartesian; })
            .addProperty("trust_radius", &OptimizationState::trust_radius)
            .addProperty("first_step", &OptimizationState::first_step)
            .addProperty("step_number", &OptimizationState::step_number)
            .addProperty("converged", &OptimizationState::converged)
          .endClass()

          .beginClass<BernyOptimizer>("BernyOptimizer")
            // Two construction shapes — molecule-only vs criteria-configured.
            .addConstructor<void (*)(const occ::core::Molecule &,
                                      const ConvergenceCriteria &)>()
            .addStaticFunction(
                "new_default",
                +[](const occ::core::Molecule &m) {
                  return new BernyOptimizer(m);
                })
            .addFunction("step", &BernyOptimizer::step)
            // Accept either a Mat3N userdata (direct from
            // compute_gradient) or a nested Lua table.
            .addFunction("update",
                         +[](BernyOptimizer *bo, double energy,
                             const lb::LuaRef &gradient) {
                           if (gradient.isUserdata()) {
                             const occ::Mat3N &g =
                                 gradient.unsafe_cast<const occ::Mat3N &>();
                             bo->update(energy, g);
                           } else {
                             bo->update(energy, table_to_mat3n(gradient));
                           }
                         })
            .addFunction("get_next_geometry",
                         +[](const BernyOptimizer *bo) {
                           return bo->get_next_geometry();
                         })
            .addProperty("is_converged", &BernyOptimizer::is_converged)
            .addProperty("current_step", &BernyOptimizer::current_step)
            .addProperty("current_energy", &BernyOptimizer::current_energy)
            .addProperty("current_trust_radius",
                         &BernyOptimizer::current_trust_radius)
            .addFunction("__tostring", +[](const BernyOptimizer *b) {
              return fmt::format("<BernyOptimizer: step={}, converged={}>",
                                 b->current_step(), b->is_converged());
            })
          .endClass()

        .endNamespace()
      .endNamespace();
}

} // namespace occ::lua_bindings
