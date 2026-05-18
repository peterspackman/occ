#include "dma_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/dma/dma.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/dma/mult.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/driver/dma_driver.h>
#include <occ/qm/wavefunction.h>

namespace sol {
template <> struct is_automagical<occ::dma::Mult> : std::false_type {};
template <> struct is_automagical<occ::dma::DMASites> : std::false_type {};
template <> struct is_automagical<occ::dma::DMAResult> : std::false_type {};
template <>
struct is_automagical<occ::dma::DMACalculator> : std::false_type {};
template <>
struct is_automagical<occ::dma::LinearMultipoleCalculator>
    : std::false_type {};
template <>
struct is_automagical<occ::driver::DMADriver> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::dma;

void register_dma_bindings(sol::state_view, sol::table &m) {
  m.new_usertype<Mult>(
      "Mult",
      sol::call_constructor,
      sol::factories([]() { return Mult{}; },
                     [](int max_rank) { return Mult(max_rank); }),
      "max_rank", &Mult::max_rank,
      "q",
      sol::property(
          [](const Mult &mp, sol::this_state s) {
            return mp.q;
          },
          [](Mult &mp, const sol::table &t) { mp.q = table_to_vecx(t); }),
      "num_components", &Mult::num_components,
      "to_string", &Mult::to_string,
      "get_multipole",
      [](const Mult &mp, int l, int mm) { return mp.get_multipole(l, mm); },
      "get_component",
      [](const Mult &mp, const std::string &name) {
        return mp.get_component(name);
      },
      sol::meta_function::to_string, [](const Mult &mp) {
        return fmt::format("<Mult max_rank={} components={}>", mp.max_rank,
                           mp.num_components());
      });

  m.set_function("component_name_to_lm", &Mult::component_name_to_lm);

  m.new_usertype<DMASettings>(
      "DMASettings",
      sol::call_constructor, sol::factories([]() { return DMASettings{}; }),
      "max_rank", &DMASettings::max_rank,
      "big_exponent", &DMASettings::big_exponent,
      "include_nuclei", &DMASettings::include_nuclei,
      sol::meta_function::to_string, [](const DMASettings &s) {
        return fmt::format("<DMASettings max_rank={} big_exponent={:.2f} "
                           "include_nuclei={}>",
                           s.max_rank, s.big_exponent,
                           s.include_nuclei ? "true" : "false");
      });

  m.new_usertype<DMAResult>(
      "DMAResult",
      sol::call_constructor, sol::factories([]() { return DMAResult{}; }),
      "max_rank", &DMAResult::max_rank,
      "multipoles",
      sol::readonly_property(
          [](const DMAResult &r) { return sol::as_table(r.multipoles); }),
      sol::meta_function::to_string, [](const DMAResult &r) {
        return fmt::format("<DMAResult max_rank={} num_sites={}>",
                           r.max_rank, r.multipoles.size());
      });

  m.new_usertype<DMASites>(
      "DMASites",
      sol::call_constructor, sol::factories([]() { return DMASites{}; }),
      "size", &DMASites::size,
      "num_atoms", &DMASites::num_atoms,
      "atoms",
      sol::readonly_property(
          [](const DMASites &s) { return sol::as_table(s.atoms); }),
      "name",
      sol::readonly_property(
          [](const DMASites &s) { return sol::as_table(s.name); }),
      "positions",
      sol::readonly_property([](const DMASites &s, sol::this_state st) {
        return mat_to_table(st, s.positions);
      }),
      "atom_indices",
      sol::readonly_property([](const DMASites &s, sol::this_state st) {
        return vec_to_table(st, s.atom_indices);
      }),
      "radii",
      sol::readonly_property([](const DMASites &s, sol::this_state st) {
        return vec_to_table(st, s.radii);
      }),
      "limits",
      sol::readonly_property([](const DMASites &s, sol::this_state st) {
        return vec_to_table(st, s.limits);
      }),
      sol::meta_function::to_string, [](const DMASites &s) {
        return fmt::format("<DMASites num_sites={} num_atoms={}>", s.size(),
                           s.num_atoms());
      });

  m.new_usertype<DMACalculator>(
      "DMACalculator",
      sol::call_constructor,
      sol::constructors<DMACalculator(const occ::qm::Wavefunction &)>(),
      "update_settings", &DMACalculator::update_settings,
      "settings", &DMACalculator::settings,
      "set_radius_for_element", &DMACalculator::set_radius_for_element,
      "set_limit_for_element", &DMACalculator::set_limit_for_element,
      "sites", &DMACalculator::sites,
      "compute_multipoles", &DMACalculator::compute_multipoles,
      "compute_total_multipoles", &DMACalculator::compute_total_multipoles,
      sol::meta_function::to_string, [](const DMACalculator &c) {
        return fmt::format("<DMACalculator num_sites={} max_rank={}>",
                           c.sites().size(), c.settings().max_rank);
      });

  m.new_usertype<LinearDMASettings>(
      "LinearDMASettings",
      sol::call_constructor,
      sol::factories([]() { return LinearDMASettings{}; }),
      "max_rank", &LinearDMASettings::max_rank,
      "include_nuclei", &LinearDMASettings::include_nuclei,
      "use_slices", &LinearDMASettings::use_slices,
      "tolerance", &LinearDMASettings::tolerance,
      "default_radius", &LinearDMASettings::default_radius,
      "hydrogen_radius", &LinearDMASettings::hydrogen_radius);

  m.new_usertype<LinearMultipoleCalculator>(
      "LinearMultipoleCalculator",
      sol::call_constructor,
      sol::factories(
          [](const occ::qm::Wavefunction &w) {
            return LinearMultipoleCalculator(w, LinearDMASettings{});
          },
          [](const occ::qm::Wavefunction &w, const LinearDMASettings &s) {
            return LinearMultipoleCalculator(w, s);
          }),
      "calculate", &LinearMultipoleCalculator::calculate);

  m.new_usertype<occ::driver::DMAConfig>(
      "DMAConfig",
      sol::call_constructor,
      sol::factories([]() { return occ::driver::DMAConfig{}; }),
      "wavefunction_filename", &occ::driver::DMAConfig::wavefunction_filename,
      "punch_filename", &occ::driver::DMAConfig::punch_filename,
      "settings", &occ::driver::DMAConfig::settings,
      "write_punch", &occ::driver::DMAConfig::write_punch);

  m.new_usertype<occ::driver::DMADriver::DMAOutput>(
      "DMAOutput", sol::no_constructor,
      "result", &occ::driver::DMADriver::DMAOutput::result,
      "sites", &occ::driver::DMADriver::DMAOutput::sites);

  m.new_usertype<occ::driver::DMADriver>(
      "DMADriver",
      sol::call_constructor,
      sol::factories(
          []() { return occ::driver::DMADriver{}; },
          [](const occ::driver::DMAConfig &c) {
            return occ::driver::DMADriver(c);
          }),
      "set_config", &occ::driver::DMADriver::set_config,
      "config", &occ::driver::DMADriver::config,
      "run",
      sol::overload(
          [](occ::driver::DMADriver &d) { return d.run(); },
          [](occ::driver::DMADriver &d, const occ::qm::Wavefunction &w) {
            return d.run(w);
          }));

  m.set_function("dma_generate_punch_file",
                 &occ::driver::DMADriver::generate_punch_file);
  m.set_function("dma_write_punch_file",
                 &occ::driver::DMADriver::write_punch_file);
}

} // namespace occ::lua_bindings
