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

// BondCoordinate / AngleCoordinate / DihedralCoordinate all have
// `operator()(const Mat3N &)` returning a double — sol2's automagic
// `call_operator` enrollment would otherwise instantiate the Mat3N
// container probe and trigger the void cascade.
namespace sol {
template <>
struct is_automagical<occ::opt::BondCoordinate> : std::false_type {};
template <>
struct is_automagical<occ::opt::AngleCoordinate> : std::false_type {};
template <>
struct is_automagical<occ::opt::DihedralCoordinate> : std::false_type {};
template <>
struct is_automagical<occ::opt::InternalCoordinates> : std::false_type {};
template <>
struct is_automagical<occ::opt::OptimizationState> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::opt;

void register_opt_bindings(sol::state_view, sol::table &occ_module) {
  // Python uses a submodule (`occpy.opt.X`); mirror that as a nested Lua
  // table.
  sol::table opt = occ_module.create_named("opt");

  opt.new_enum<BondCoordinate::Type>(
      "BondCoordinateType", {{"COVALENT", BondCoordinate::Type::COVALENT},
                             {"VDW", BondCoordinate::Type::VDW}});

  opt.new_usertype<BondCoordinate>(
      "BondCoordinate",
      sol::call_constructor,
      sol::factories(
          [](int i, int j) {
            return BondCoordinate{i, j, BondCoordinate::Type::COVALENT};
          },
          [](int i, int j, BondCoordinate::Type t) {
            return BondCoordinate{i, j, t};
          }),
      "i", sol::readonly(&BondCoordinate::i),
      "j", sol::readonly(&BondCoordinate::j),
      "bond_type", sol::readonly(&BondCoordinate::bond_type),
      "value",
      [](const BondCoordinate &b, const sol::table &coords) {
        return b(table_to_mat3n(coords));
      },
      "gradient",
      [](const BondCoordinate &b, const sol::table &coords, sol::this_state s) {
        return b.gradient(table_to_mat3n(coords));
      },
      sol::meta_function::to_string, [](const BondCoordinate &b) {
        const char *type = (b.bond_type == BondCoordinate::Type::COVALENT)
                               ? "COVALENT"
                               : "VDW";
        return fmt::format("<BondCoordinate({}, {}, {})>", b.i, b.j, type);
      });

  opt.new_usertype<AngleCoordinate>(
      "AngleCoordinate",
      sol::call_constructor, sol::constructors<AngleCoordinate(int, int, int)>(),
      "i", &AngleCoordinate::i,
      "j", &AngleCoordinate::j,
      "k", &AngleCoordinate::k,
      "value",
      [](const AngleCoordinate &a, const sol::table &coords) {
        return a(table_to_mat3n(coords));
      },
      "gradient",
      [](const AngleCoordinate &a, const sol::table &coords, sol::this_state s) {
        return a.gradient(table_to_mat3n(coords));
      },
      sol::meta_function::to_string, [](const AngleCoordinate &a) {
        return fmt::format("<AngleCoordinate({}, {}, {})>", a.i, a.j, a.k);
      });

  opt.new_usertype<DihedralCoordinate>(
      "DihedralCoordinate",
      sol::call_constructor,
      sol::constructors<DihedralCoordinate(int, int, int, int)>(),
      "i", &DihedralCoordinate::i,
      "j", &DihedralCoordinate::j,
      "k", &DihedralCoordinate::k,
      "l", &DihedralCoordinate::l,
      "value",
      [](const DihedralCoordinate &d, const sol::table &coords) {
        return d(table_to_mat3n(coords));
      },
      "gradient",
      [](const DihedralCoordinate &d, const sol::table &coords,
         sol::this_state s) {
        return d.gradient(table_to_mat3n(coords));
      },
      sol::meta_function::to_string, [](const DihedralCoordinate &d) {
        return fmt::format("<DihedralCoordinate({}, {}, {}, {})>", d.i, d.j,
                           d.k, d.l);
      });

  opt.new_usertype<InternalCoordinates::Options>(
      "InternalCoordinatesOptions",
      sol::call_constructor,
      sol::factories([]() { return InternalCoordinates::Options{}; }),
      "include_dihedrals", &InternalCoordinates::Options::include_dihedrals,
      "superweak_dihedrals",
      &InternalCoordinates::Options::superweak_dihedrals);

  opt.new_usertype<InternalCoordinates>(
      "InternalCoordinates",
      sol::call_constructor,
      sol::factories(
          [](const occ::core::Molecule &m) {
            return InternalCoordinates(m, InternalCoordinates::Options{true,
                                                                       false});
          },
          [](const occ::core::Molecule &m,
             const InternalCoordinates::Options &opts) {
            return InternalCoordinates(m, opts);
          }),
      "bonds",
      [](const InternalCoordinates &ic) {
        return sol::as_table(ic.bonds());
      },
      "angles",
      [](const InternalCoordinates &ic) {
        return sol::as_table(ic.angles());
      },
      "dihedrals",
      [](const InternalCoordinates &ic) {
        return sol::as_table(ic.dihedrals());
      },
      "weights",
      [](const InternalCoordinates &ic, sol::this_state s) {
        return ic.weights();
      },
      "size", &InternalCoordinates::size,
      sol::meta_function::length, &InternalCoordinates::size,
      "to_vector",
      [](const InternalCoordinates &ic, const sol::table &positions,
         sol::this_state s) {
        return ic.to_vector(table_to_mat3n(positions));
      },
      "to_vector_with_template",
      [](const InternalCoordinates &ic, const sol::table &positions,
         const sol::table &template_q, sol::this_state s) {
        return vec_to_table(
            s, ic.to_vector_with_template(table_to_mat3n(positions),
                                            table_to_vecx(template_q)));
      },
      "wilson_b_matrix",
      [](const InternalCoordinates &ic, const sol::table &positions,
         sol::this_state s) {
        return ic.wilson_b_matrix(table_to_mat3n(positions));
      },
      sol::meta_function::to_string, [](const InternalCoordinates &ic) {
        return fmt::format(
            "<InternalCoordinates: {} bonds, {} angles, {} dihedrals>",
            ic.bonds().size(), ic.angles().size(), ic.dihedrals().size());
      });

  opt.set_function(
      "transform_step_to_cartesian",
      [](const sol::table &internal_step, const InternalCoordinates &coords,
         const sol::table &positions, const sol::table &B_inv,
         sol::this_state s) {
        Mat Binv(static_cast<int>(B_inv.size()),
                  static_cast<int>(B_inv.get<sol::table>(1).size()));
        for (int i = 0; i < Binv.rows(); ++i) {
          sol::table row = B_inv.get<sol::table>(i + 1);
          for (int j = 0; j < Binv.cols(); ++j) {
            Binv(i, j) = row.get<double>(j + 1);
          }
        }
        return mat_to_table(s, occ::opt::transform_step_to_cartesian(
                                    table_to_vecx(internal_step), coords,
                                    table_to_mat3n(positions), Binv));
      });

  opt.new_usertype<OptPoint>(
      "OptPoint",
      sol::call_constructor,
      sol::factories(
          []() { return OptPoint{}; },
          [](const sol::table &q, double energy, const sol::table &gradient) {
            return OptPoint{table_to_vecx(q), energy, table_to_vecx(gradient)};
          }),
      "q",
      sol::property(
          [](const OptPoint &p, sol::this_state s) {
            return p.q;
          },
          [](OptPoint &p, const sol::table &t) { p.q = table_to_vecx(t); }),
      "E", &OptPoint::E,
      "g",
      sol::property(
          [](const OptPoint &p, sol::this_state s) {
            return p.g;
          },
          [](OptPoint &p, const sol::table &t) { p.g = table_to_vecx(t); }),
      sol::meta_function::to_string, [](const OptPoint &p) {
        return fmt::format("<OptPoint: E={:.8f}, |q|={}, |g|={:.6f}>", p.E,
                           p.q.size(), p.g.size() > 0 ? p.g.norm() : 0.0);
      });

  opt.new_usertype<ConvergenceCriteria>(
      "ConvergenceCriteria",
      sol::call_constructor,
      sol::factories([]() { return ConvergenceCriteria{}; }),
      "gradient_max", &ConvergenceCriteria::gradient_max,
      "gradient_rms", &ConvergenceCriteria::gradient_rms,
      "step_max", &ConvergenceCriteria::step_max,
      "step_rms", &ConvergenceCriteria::step_rms);

  // OptimizationState exposes many Eigen-backed fields — wrap each via a
  // readonly property so we never hand sol2 a raw member pointer.
  opt.new_usertype<OptimizationState>(
      "OptimizationState", sol::no_constructor,
      "positions",
      sol::readonly_property([](const OptimizationState &o, sol::this_state s) {
        return o.positions;
      }),
      "energy", &OptimizationState::energy,
      "gradient_cartesian",
      sol::readonly_property([](const OptimizationState &o, sol::this_state s) {
        return o.gradient_cartesian;
      }),
      "trust_radius", &OptimizationState::trust_radius,
      "first_step", &OptimizationState::first_step,
      "step_number", &OptimizationState::step_number,
      "converged", &OptimizationState::converged);

  opt.new_usertype<BernyOptimizer>(
      "BernyOptimizer",
      sol::call_constructor,
      sol::factories(
          [](const occ::core::Molecule &m) { return BernyOptimizer(m); },
          [](const occ::core::Molecule &m, const ConvergenceCriteria &c) {
            return BernyOptimizer(m, c);
          }),
      "step", &BernyOptimizer::step,
      "update",
      [](BernyOptimizer &bo, double energy, const sol::table &gradient) {
        bo.update(energy, table_to_mat3n(gradient));
      },
      // Returns a core::Molecule, not raw Mat3N positions.
      // Returns a core::Molecule, not raw Mat3N positions.
      "get_next_geometry",
      [](const BernyOptimizer &bo) { return bo.get_next_geometry(); },
      "is_converged", &BernyOptimizer::is_converged,
      "current_step", &BernyOptimizer::current_step,
      "current_energy", &BernyOptimizer::current_energy,
      "current_trust_radius", &BernyOptimizer::current_trust_radius,
      sol::meta_function::to_string, [](const BernyOptimizer &b) {
        return fmt::format("<BernyOptimizer: step={}, converged={}>",
                           b.current_step(), b.is_converged());
      });
}

} // namespace occ::lua_bindings
